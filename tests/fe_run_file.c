/*
 * fe_run_file.c -- Stream a raw float32 PCM file through fe_run and write the
 *                 result directly as a 48 kHz PCM16 WAV. .f32 output is
 *                 unplayable and just clutters results/, so we emit WAV
 *                 only -- pass any extension; the file is always a valid WAV.
 *
 * Usage: fe_run_file <weights.q8> <input.f32> <output.wav> <num_frames>
 *   weights.q8 : q8 weights blob
 *   input.f32   : num_frames * FE_FRAME_SIZE float32 samples (48 kHz)
 *   output.wav  : enhanced audio as PCM16 WAV @ 48 kHz mono
 */
#include "fe.h"
#include "fe_profile.h"
#include "qgemm/qgemm_dispatch.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(_WIN32)
#  include <windows.h>
#else
#  include <time.h>
#  include <pthread.h>
#  include <sched.h>
#  if defined(__APPLE__)
#    include <pthread/qos.h>
#  endif
#endif

static int cmp_dbl_g(const void *a, const void *b) {
    double da = *(const double *)a, db = *(const double *)b;
    return (da > db) - (da < db);
}

/* Write a single-channel 48 kHz PCM16 WAV. Clip fp32 to ±1 then scale to
 * int16. Streams to disk in one call; no intermediate buffer past the
 * int16 conversion.
 *
 * Every fwrite return value is checked, and the file is fflushed before
 * fclose. Without these checks a global -falign-functions=32 build
 * silently truncated the WAV by ~1 frame (684 byte) — apparently the C
 * runtime's stdio buffer flush could short-write under that codegen
 * variant. Defensive return-value checking + explicit flush prevents
 * silent truncation regardless of codegen perturbations. */
static int write_wav_pcm16(const char *path, const float *samples,
                           size_t n_samples, uint32_t sample_rate) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    const uint16_t channels    = 1;
    const uint16_t bits        = 16;
    const uint16_t block_align = channels * (bits / 8);
    const uint32_t byte_rate   = sample_rate * block_align;
    const uint32_t data_bytes  = (uint32_t)(n_samples * sizeof(int16_t));
    const uint32_t riff_size   = 36 + data_bytes;

#define WW(buf, sz, n) do {                                                 \
    if (fwrite((buf), (sz), (n), f) != (size_t)(n)) {                       \
        fprintf(stderr, "wav write short at line %d\n", __LINE__);          \
        fclose(f); return -3;                                               \
    }                                                                       \
} while (0)

    WW("RIFF", 1, 4);
    WW(&riff_size, 4, 1);
    WW("WAVEfmt ", 1, 8);
    uint32_t fmt_size = 16;          WW(&fmt_size,    4, 1);
    uint16_t fmt_pcm  = 1;           WW(&fmt_pcm,     2, 1);
    WW(&channels, 2, 1);
    WW(&sample_rate, 4, 1);
    WW(&byte_rate,   4, 1);
    WW(&block_align, 2, 1);
    WW(&bits,        2, 1);
    WW("data", 1, 4);
    WW(&data_bytes, 4, 1);

    int16_t *pcm = (int16_t *)malloc(data_bytes);
    if (!pcm) { fclose(f); return -2; }
    for (size_t i = 0; i < n_samples; ++i) {
        float x = samples[i];
        if (x >  1.0f) x =  1.0f;
        if (x < -1.0f) x = -1.0f;
        pcm[i] = (int16_t)(x * 32767.0f);
    }

    /* Single PCM body write — chunked retry loop in case the C runtime's
     * stdio buffer short-writes. */
    {
        size_t written = 0;
        while (written < data_bytes) {
            size_t got = fwrite((const char *)pcm + written, 1,
                                data_bytes - written, f);
            if (got == 0) {
                fprintf(stderr, "wav pcm write stalled at %zu / %u\n",
                        written, data_bytes);
                free(pcm); fclose(f); return -4;
            }
            written += got;
        }
    }
    free(pcm);

    if (fflush(f) != 0) {
        fprintf(stderr, "wav fflush failed\n");
        fclose(f); return -5;
    }
    if (fclose(f) != 0) {
        fprintf(stderr, "wav fclose failed\n");
        return -6;
    }
#undef WW
    return 0;
}

static int load_file(const char *path, void **buf, size_t *sz) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "open %s failed\n", path); return -1; }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    void *p = malloc((size_t)n);
    if (!p || fread(p, 1, (size_t)n, f) != (size_t)n) {
        free(p); fclose(f); return -1;
    }
    fclose(f);
    *buf = p; *sz = (size_t)n;
    return 0;
}

static double sec_now(void) {
#if defined(_WIN32)
    static LARGE_INTEGER freq = {0};
    if (!freq.QuadPart) QueryPerformanceFrequency(&freq);
    LARGE_INTEGER ctr; QueryPerformanceCounter(&ctr);
    return (double)ctr.QuadPart / (double)freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
#endif
}

/* Elevate this benchmark thread to an audio-callback-like priority -- the
 * representative deployment condition. Real integrations run fe_run on a
 * high-priority audio thread (WASAPI / CoreAudio / AAudio all do); measuring
 * at *normal desktop priority* makes the tail (p99/peak) look far worse than
 * production, because the Win11 / Intel Thread Director scheduler parks a
 * normal-priority thread on an E-core mid-stream (Gracemont runs the VNNI
 * kernel ~3.6x slower than a Raptor Cove P-core). High priority biases the
 * scheduler to keep the thread on a P-core and cuts preemption, flattening
 * p99 to ~1.3-1.4x p50 -- which is what a real audio thread already sees.
 *
 * On by default (the realistic condition); set FE_RT=0 to measure the
 * pessimistic normal-priority envelope for comparison. Best-effort: any
 * failure (no privilege, unsupported) is a harmless no-op -- the engine is
 * byte-identical either way, only the jitter envelope differs.
 *
 * NOTE: this lives in the *benchmark harness*, not the library. fe itself
 * stays an OS-neutral C ABI (4 public functions); thread scheduling is the
 * integrator's concern. See docs/optimizations/latency-and-measurement.md. */
static int realtime_thread_enabled(void) {
    const char *e = getenv("FE_RT");
    return !(e && e[0] == '0');   /* default on; FE_RT=0 disables */
}

/* FE_QOS selects the scheduling class explicitly, which is what lets one
 * harness measure both deployment placements on a heterogeneous CPU:
 *   hi  (default) -- USER_INTERACTIVE, the audio-thread class; P-core.
 *   lo            -- BACKGROUND, which macOS confines to E-cores. This is the
 *                    "pace on a small core" placement, not a fallback.
 *   off           -- leave the inherited class alone (same as FE_RT=0).
 * Only meaningful on Apple Silicon; ignored elsewhere. */
static const char *qos_pref(void) {
    const char *e = getenv("FE_QOS");
    return e ? e : "hi";
}

static const char *apply_realtime_thread(void) {
#if defined(__APPLE__)
    {
        const char *q = qos_pref();
        if (q[0] == 'l') {   /* lo -> E-core */
            if (pthread_set_qos_class_self_np(QOS_CLASS_BACKGROUND, 0) == 0)
                return "background(QOS_BACKGROUND, E-core)";
            return "normal(qos failed)";
        }
        if (q[0] == 'o') return "normal(QOS untouched)";
    }
#endif
    if (!realtime_thread_enabled()) return "normal";
#if defined(_WIN32)
    /* TIME_CRITICAL = base priority 15, the level WASAPI/ASIO audio render
     * threads run at -- the faithful deployment condition. (MMCSS "Pro Audio"
     * is what real apps use but needs avrt.lib; TIME_CRITICAL is equivalent
     * for steering off E-cores without an extra link dependency.) */
    if (SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL))
        return "realtime(THREAD_PRIORITY_TIME_CRITICAL)";
    return "normal(elevation failed)";
#elif defined(__APPLE__)
    /* QoS USER_INTERACTIVE keeps the thread on M-series P-cores; no priv. */
    if (pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0) == 0)
        return "realtime(QOS_USER_INTERACTIVE)";
    return "normal(qos failed)";
#elif defined(__linux__)
    /* SCHED_FIFO needs CAP_SYS_NICE / RT-capable; harmless no-op if denied. */
    struct sched_param sp;
    sp.sched_priority = (sched_get_priority_min(SCHED_FIFO) +
                         sched_get_priority_max(SCHED_FIFO)) / 2;
    if (pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp) == 0)
        return "realtime(SCHED_FIFO)";
    return "normal(no RT privilege)";
#else
    return "normal(unsupported)";
#endif
}

int main(int argc, char **argv) {
    if (argc != 5 && argc != 6) {
        fprintf(stderr,
            "usage: %s <weights.q8> <input.f32> <output.wav> <num_frames> [tier]\n"
            "  tier (optional): neon | dotprod | i8mm | avx2 | avxvnni | avx512vnni\n",
            argv[0]);
        return 1;
    }
    int n_frames = atoi(argv[4]);
    if (n_frames <= 0) { fprintf(stderr, "bad n_frames\n"); return 1; }

    if (argc == 6) {
        if (fe_qgemm_force_tier(argv[5]) != 0) {
            fprintf(stderr, "tier '%s' not available on this build/CPU\n", argv[5]);
            return 2;
        }
    }

    void *wbuf; size_t wsz;
    if (load_file(argv[1], &wbuf, &wsz) != 0) return 1;

    if (fe_init(wbuf, (int)wsz) != 0) {
        fprintf(stderr, "fe_init failed\n");
        return 1;
    }

    void *ibuf; size_t isz;
    if (load_file(argv[2], &ibuf, &isz) != 0) return 1;
    size_t need = (size_t)n_frames * FE_FRAME_SIZE * sizeof(float);
    if (isz < need) {
        fprintf(stderr, "input too small: have %zu, need %zu\n", isz, need);
        return 1;
    }

    float *out = (float *)malloc(need);
    const float *frames = (const float *)ibuf;

    /* Per-frame timing -- keep all frames so startup/warm-cache effects can
     * be separated from the later window. */
    double *ft = (double *)malloc((size_t)n_frames * sizeof(double));
    const char *rt_mode = apply_realtime_thread();   /* default on; FE_RT=0 off */
    fprintf(stderr, "thread priority: %s\n", rt_mode);
    /* FE_PACE=1 processes one frame per frame-period of wall clock instead
     * of racing through the file. Throughput numbers are unchanged (ft[] still
     * times compute only), but the power draw becomes the deployment one: a
     * real stream delivers a frame every 6.67 ms, so racing to idle at peak
     * clock measures an operating point that never occurs in an audio callback.
     * Only meaningful when the run is actually real-time capable. */
    const char *pace_env = getenv("FE_PACE");
    const int pace = pace_env && pace_env[0] == '1';
    const double period = (double)FE_FRAME_SIZE / (double)FE_SAMPLE_RATE;
    int late_frames = 0;

    double t0 = sec_now();
    for (int t = 0; t < n_frames; ++t) {
        double s = sec_now();
        fe_run(frames + (size_t)t * FE_FRAME_SIZE,
               out    + (size_t)t * FE_FRAME_SIZE);
        ft[t] = sec_now() - s;
        if (pace) {
            double deadline = t0 + (double)(t + 1) * period;
            double slack = deadline - sec_now();
            if (slack > 0) {
#if defined(_WIN32)
                Sleep((DWORD)(slack * 1000.0));
#else
                struct timespec ts;
                ts.tv_sec  = (time_t)slack;
                ts.tv_nsec = (long)((slack - (double)ts.tv_sec) * 1e9);
                nanosleep(&ts, NULL);
#endif
            } else {
                ++late_frames;   /* missed the deadline: not real-time here */
            }
        }
    }
    double elapsed = sec_now() - t0;
    if (pace)
        fprintf(stderr, "pacing: one frame per %.3f ms, %d/%d frames missed "
                        "the deadline (%.3f%%)\n",
                period * 1000.0, late_frames, n_frames,
                100.0 * (double)late_frames / (double)n_frames);
    double audio_s = (double)n_frames * FE_FRAME_SIZE / FE_SAMPLE_RATE;

    fprintf(stderr,
            "fe: %d frames (%.2f s audio) in %.1f ms  "
            "(%.3f ms/frame, RTF %.4f)\n",
            n_frames, audio_s, elapsed * 1000.0,
            elapsed / n_frames * 1000.0,
            elapsed / audio_s);

    /* Window analysis. Report first 100, next 100, and later frames
     * separately. Activation scales are per-frame dynamic and never freeze;
     * this split is only a startup/warm-cache diagnostic. */
    if (n_frames >= 300) {
        double s_early = 0, s_mid = 0, s_late = 0;
        for (int t = 0; t < 100; ++t)             s_early += ft[t];
        for (int t = 100; t < 200; ++t)           s_mid   += ft[t];
        for (int t = 200; t < n_frames; ++t)      s_late  += ft[t];
        fprintf(stderr,
                "window split: [0..99] %.3f ms/frame   [100..199] %.3f ms/frame   "
                "[200..%d] %.3f ms/frame\n",
                s_early / 100.0 * 1000.0,
                s_mid   / 100.0 * 1000.0,
                n_frames - 1,
                s_late  / (double)(n_frames - 200) * 1000.0);
        fprintf(stderr,
                "early-vs-late delta: %.3f ms/frame x 100 frames = %.1f ms total "
                "(%.2f%% of run)\n",
                (s_early / 100.0 - s_late / (double)(n_frames - 200)) * 1000.0,
                (s_early - s_late / (double)(n_frames - 200) * 100.0) * 1000.0,
                100.0 * (s_early - s_late / (double)(n_frames - 200) * 100.0) / elapsed);
    }
    /* Per-frame peak / percentile analysis (later window only -- skip first
     * 200 frames to exclude startup/warm-cache effects). */
    if (n_frames >= 300) {
        int n_ss = n_frames - 200;
        double *ss = (double *)malloc((size_t)n_ss * sizeof(double));
        for (int t = 0; t < n_ss; ++t) ss[t] = ft[200 + t];
        qsort(ss, (size_t)n_ss, sizeof(double), cmp_dbl_g);
        double frame_period_s = (double)FE_FRAME_SIZE / FE_SAMPLE_RATE;
        double p50  = ss[(int)(n_ss * 0.50)];
        double p95  = ss[(int)(n_ss * 0.95)];
        double p99  = ss[(int)(n_ss * 0.99)];
        double p999 = ss[(int)(n_ss * 0.999)];
        double pmax = ss[n_ss - 1];
        fprintf(stderr,
                "per-frame (steady state, %d frames):\n"
                "  p50  = %.3f ms (RTF %.4f)\n"
                "  p95  = %.3f ms (RTF %.4f)\n"
                "  p99  = %.3f ms (RTF %.4f)\n"
                "  p99.9= %.3f ms (RTF %.4f)\n"
                "  peak = %.3f ms (RTF %.4f)\n"
                "  frame budget = %.3f ms (%d samples @ %d Hz)\n",
                n_ss,
                p50 * 1000,  p50 / frame_period_s,
                p95 * 1000,  p95 / frame_period_s,
                p99 * 1000,  p99 / frame_period_s,
                p999 * 1000, p999 / frame_period_s,
                pmax * 1000, pmax / frame_period_s,
                frame_period_s * 1000, FE_FRAME_SIZE, FE_SAMPLE_RATE);
        free(ss);
    }
    free(ft);

    fe_profile_dump(n_frames);

    /* Emit a 48 kHz PCM16 WAV -- playable directly, no manual conversion. */
    if (write_wav_pcm16(argv[3], out,
                        (size_t)n_frames * FE_FRAME_SIZE,
                        FE_SAMPLE_RATE) != 0) {
        fprintf(stderr, "wav write failed: %s\n", argv[3]);
    }
    free(out);
    fe_free();
    return 0;
}
