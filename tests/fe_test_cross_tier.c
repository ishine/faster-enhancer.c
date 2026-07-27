// fe_test_cross_tier -- Bit-identity test harness across all GEMM tiers
// available on the host CPU/build. Loads weights, runs a fixed number of
// frames per tier with the same input, and compares the final float waveform
// bit-for-bit against the highest-tier reference in the same architecture
// family. SNR is printed as a diagnostic, but any byte drift fails the test.
//
// Usage:  fe_test_cross_tier <weights.q8> <input.f32> <num_frames>
#include "fe.h"
#include "qgemm/qgemm_dispatch.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int load_file(const char *path, void **buf, size_t *sz) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
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

static double snr_db(const float *ref, const float *test, size_t n) {
    double sig = 0.0, noise = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double r = ref[i], d = ref[i] - test[i];
        sig   += r * r;
        noise += d * d;
    }
    if (noise <= 0.0) return 300.0;
    return 10.0 * log10(sig / noise);
}

static int run_one_tier(const char *tier_name,
                        void *wbuf, int wsz,
                        const float *frames, int n_frames,
                        float *out, size_t out_n) {
    fe_free();
    if (fe_qgemm_force_tier(tier_name) != 0) return 0;  // not available
    if (fe_init(wbuf, wsz) != 0) return -1;

    for (int t = 0; t < n_frames; ++t) {
        fe_run(frames + (size_t)t * FE_FRAME_SIZE,
               out    + (size_t)t * FE_FRAME_SIZE);
    }
    (void)out_n;
    return 1;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s <weights.q8> <input.f32> <num_frames>\n",
                argv[0]);
        return 1;
    }
    int n_frames = atoi(argv[3]);

    void *wbuf; size_t wsz;
    if (load_file(argv[1], &wbuf, &wsz) != 0) {
        fprintf(stderr, "open %s failed\n", argv[1]);
        return 1;
    }
    void *ibuf; size_t isz;
    if (load_file(argv[2], &ibuf, &isz) != 0) {
        fprintf(stderr, "open %s failed\n", argv[2]);
        return 1;
    }
    size_t need = (size_t)n_frames * FE_FRAME_SIZE * sizeof(float);
    if (isz < need) {
        fprintf(stderr, "input too small\n");
        return 1;
    }
    const float *frames = (const float *)ibuf;
    size_t out_n = (size_t)n_frames * FE_FRAME_SIZE;

    // Tiers to probe, in ascending capability. Reference is the highest one
    // that's available on the host (typically i8mm on Apple M-series, or
    // avx512vnni on a server, etc.). Lower tiers are then compared against
    // the reference.
    const char *tier_names[] = {
        "neon", "dotprod", "i8mm",
        "avx2", "avxvnni", "avx512vnni",
    };
    const int n_tiers = (int)(sizeof(tier_names) / sizeof(tier_names[0]));

    float *ref = (float *)malloc(out_n * sizeof(float));
    float *cur = (float *)malloc(out_n * sizeof(float));
    if (!ref || !cur) return 1;

    // Find the highest tier available on this host/build and use it as ref.
    const char *ref_tier_name = NULL;
    for (int i = n_tiers - 1; i >= 0; --i) {
        if (run_one_tier(tier_names[i], wbuf, (int)wsz, frames, n_frames, ref, out_n) == 1) {
            ref_tier_name = tier_names[i];
            break;
        }
    }
    if (!ref_tier_name) {
        fprintf(stderr, "no tier available\n");
        return 1;
    }
    fprintf(stderr, "reference tier: %s\n", ref_tier_name);

    int failed = 0;
    for (int i = 0; i < n_tiers; ++i) {
        if (!strcmp(tier_names[i], ref_tier_name)) continue;
        int avail = run_one_tier(tier_names[i], wbuf, (int)wsz, frames, n_frames, cur, out_n);
        if (avail == 0) {
            fprintf(stderr, "  %-12s : (unavailable)\n", tier_names[i]);
            continue;
        }
        if (avail < 0) {
            fprintf(stderr, "  %-12s : INIT FAILED\n", tier_names[i]);
            failed++;
            continue;
        }
        double snr = snr_db(ref, cur, out_n);
        int byte_id = (memcmp(ref, cur, out_n * sizeof(float)) == 0);
        fprintf(stderr, "  %-12s : SNR=%6.2f dB  bit-id=%s\n",
                tier_names[i], snr, byte_id ? "yes" : "no");
        if (!byte_id) failed++;
    }

    fe_free();
    free(ref); free(cur); free(wbuf); free(ibuf);
    return failed ? 2 : 0;
}
