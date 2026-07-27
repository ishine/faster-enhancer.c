/*
 * fe_bench_qgemm.c -- microbench the qgemm kernels and the qgemm-adjacent
 * fusion wrappers on model-shaped tensors.
 *
 * Usage:
 *   fe_bench_qgemm [tier|all] [iters] [repeats]
 *
 * On ARM, tiers are neon/dotprod/i8mm. On x86, avx2/avxvnni/avx512vnni.
 * The output is CSV so experiment scripts can aggregate medians.
 */
#include "fe_config_medium.h"
#include "fe_qgemm.h"
#include "fe_fp16.h"
#include "fe_calib.h"
#include "qgemm/arch_kernels.h"
#include "qgemm/qgemm_dispatch.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    const char *name;
    int M, N, K;
    double macs;
} BenchShape;

typedef struct {
    const char *family;
    const char *op;
    const BenchShape *shape;
    int iters;
    int repeats;
    double median_us;
    double gmac_s;
    double checksum;
} BenchResult;

typedef struct {
    int M, N, K;
    int8_t *Aq;
    int8_t *Bp;
    int8_t *aq;
    int8_t *Cq;
    int32_t *C32;
    int32_t *row_sums;
    float *A;
    float *A_KM;
    float *B_fp32;
    float *C;
    float *C2;
    float *scales_w;
    float *combined;
    float *bias;
    float *fp32_scratch;
    uint16_t *A_fp16;
    uint16_t *B_fp16;
} BenchBuf;

static volatile double g_sink;

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

static int cmp_double(const void *a, const void *b) {
    double da = *(const double *)a;
    double db = *(const double *)b;
    return (da > db) - (da < db);
}

static double median(double *v, int n) {
    qsort(v, (size_t)n, sizeof(double), cmp_double);
    return v[n / 2];
}

static void *xaligned(size_t n) {
    void *p = NULL;
    if (posix_memalign(&p, 64, (n + 63u) & ~63u) != 0) return NULL;
    memset(p, 0, n);
    return p;
}

static uint32_t rng_next(uint32_t *s) {
    *s = *s * 1664525u + 1013904223u;
    return *s;
}

static float rand_f32(uint32_t *s, float scale) {
    int v = (int)(rng_next(s) >> 8) & 0xffff;
    return ((float)v / 32768.0f - 1.0f) * scale;
}

static int8_t rand_i8(uint32_t *s) {
    int v = (int)(rng_next(s) % 255u) - 127;
    if (v < -127) v = -127;
    if (v >  127) v =  127;
    return (int8_t)v;
}

static void fill_case(BenchBuf *b, int M, int N, int K) {
    uint32_t seed = 0xC0FFEEu ^ (uint32_t)(M * 131 + N * 17 + K);
    b->M = M; b->N = N; b->K = K;

    size_t mk = (size_t)M * K;
    size_t nk = (size_t)N * K;
    size_t mn = (size_t)M * N;
    size_t packed = fe_qgemm_packed_size(K, N);

    b->Aq          = (int8_t *)xaligned(mk);
    b->Bp          = (int8_t *)xaligned(packed);
    b->aq          = (int8_t *)xaligned(mk > (size_t)M * 2 * K ? mk : (size_t)M * 2 * K);
    b->Cq          = (int8_t *)xaligned(mn);
    b->C32         = (int32_t *)xaligned(mn * sizeof(int32_t));
    b->row_sums    = (int32_t *)xaligned((size_t)N * sizeof(int32_t));
    b->A           = (float *)xaligned(mk * sizeof(float));
    b->A_KM        = (float *)xaligned(mk * sizeof(float));
    b->B_fp32      = (float *)xaligned(nk * sizeof(float));
    b->C           = (float *)xaligned(mn * sizeof(float));
    b->C2          = (float *)xaligned(mn * sizeof(float));
    b->scales_w    = (float *)xaligned((size_t)N * sizeof(float));
    b->combined    = (float *)xaligned((size_t)N * sizeof(float));
    b->bias        = (float *)xaligned((size_t)N * sizeof(float));
    b->fp32_scratch= (float *)xaligned(mn * sizeof(float));
    b->A_fp16      = (uint16_t *)xaligned(mk * sizeof(uint16_t));
    b->B_fp16      = (uint16_t *)xaligned(mk * sizeof(uint16_t));

    if (!b->Aq || !b->Bp || !b->aq || !b->Cq || !b->C32 || !b->row_sums ||
        !b->A || !b->A_KM || !b->B_fp32 || !b->C || !b->C2 || !b->scales_w ||
        !b->combined || !b->bias || !b->fp32_scratch || !b->A_fp16 || !b->B_fp16) {
        fprintf(stderr, "allocation failed\n");
        exit(1);
    }

    for (size_t i = 0; i < mk; ++i) {
        b->A[i] = rand_f32(&seed, 1.25f);
        b->Aq[i] = rand_i8(&seed);
    }
    for (int k = 0; k < K; ++k) {
        for (int m = 0; m < M; ++m) b->A_KM[(size_t)k * M + m] = b->A[(size_t)m * K + k];
    }
    for (size_t i = 0; i < nk; ++i) b->B_fp32[i] = rand_f32(&seed, 0.75f);
    for (int n = 0; n < N; ++n) {
        b->combined[n] = 0.0005f + 0.00001f * (float)((n * 13) & 31);
        b->bias[n] = rand_f32(&seed, 0.2f);
    }
    for (size_t i = 0; i < mn; ++i) b->C[i] = rand_f32(&seed, 0.1f);

    fe_qgemm_pack_W(b->B_fp32, N, K, b->Bp, b->scales_w);
    fe_qgemm_compute_row_sums(b->Bp, N, K, b->row_sums);
    if (fe_qgemm_ops.tier == FE_QGEMM_TIER_ARM_I8MM && (K & 7) == 0) {
        fe_qgemm_repack_i8mm(b->Bp, N, K);
    }
    fe_fp16_pack_buf(b->A_fp16, b->A_KM, (int)mk);
    fe_fp16_pack_buf(b->B_fp16, b->A, (int)mk);
}

static void free_case(BenchBuf *b) {
    free(b->Aq); free(b->Bp); free(b->aq); free(b->Cq); free(b->C32);
    free(b->row_sums); free(b->A); free(b->A_KM); free(b->B_fp32);
    free(b->C); free(b->C2); free(b->scales_w); free(b->combined);
    free(b->bias); free(b->fp32_scratch); free(b->A_fp16); free(b->B_fp16);
    memset(b, 0, sizeof(*b));
}

static double checksum_f32(const float *x, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i += 97) s += x[i];
    return s;
}

static double checksum_i32(const int32_t *x, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i += 97) s += (double)x[i];
    return s;
}

static double checksum_i8(const int8_t *x, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i += 97) s += (double)x[i];
    return s;
}

static void bench_int32_k20(const BenchBuf *b) {
#if defined(__aarch64__) || defined(__arm64__) || defined(_M_ARM64)
    if (fe_qgemm_ops.tier == FE_QGEMM_TIER_ARM_NEON)
        qgemm_neon_int32_k20(b->M, b->N, b->Aq, b->Bp, b->C32, b->N);
    else
        qgemm_dotprod_int32_k20(b->M, b->N, b->Aq, b->Bp, b->C32, b->N);
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    if (fe_qgemm_ops.tier == FE_QGEMM_TIER_X86_AVX512_VNNI)
        qgemm_avx512vnni_int32_k20(b->M, b->N, b->Aq, b->Bp, b->C32, b->N);
    else if (fe_qgemm_ops.tier == FE_QGEMM_TIER_X86_AVX_VNNI)
        qgemm_avxvnni_int32_k20(b->M, b->N, b->Aq, b->Bp, b->C32, b->N);
    else
        qgemm_avx2_int32_k20(b->M, b->N, b->Aq, b->Bp, b->C32, b->N);
#endif
}

static void bench_acc(const BenchBuf *b) {
#if defined(__aarch64__) || defined(__arm64__) || defined(_M_ARM64)
    if (fe_qgemm_ops.tier == FE_QGEMM_TIER_ARM_I8MM)
        qgemm_i8mm_fp32_fused_acc(b->M, b->N, b->K, b->Aq, b->Bp, b->combined, b->bias, b->C, b->N, b->C32);
    else if (fe_qgemm_ops.tier == FE_QGEMM_TIER_ARM_DOTPROD)
        qgemm_dotprod_fp32_fused_acc(b->M, b->N, b->K, b->Aq, b->Bp, b->combined, b->bias, b->C, b->N, b->C32);
    else
        qgemm_neon_fp32_fused_acc(b->M, b->N, b->K, b->Aq, b->Bp, b->combined, b->bias, b->C, b->N, b->C32);
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    if (fe_qgemm_ops.tier == FE_QGEMM_TIER_X86_AVX512_VNNI)
        qgemm_avx512vnni_fp32_fused_acc(b->M, b->N, b->K, b->Aq, b->Bp, b->combined, b->bias, b->C, b->N, b->C32);
    else if (fe_qgemm_ops.tier == FE_QGEMM_TIER_X86_AVX_VNNI)
        qgemm_avxvnni_fp32_fused_acc(b->M, b->N, b->K, b->Aq, b->Bp, b->combined, b->bias, b->C, b->N, b->C32);
    else
        qgemm_avx2_fp32_fused_acc(b->M, b->N, b->K, b->Aq, b->Bp, b->combined, b->bias, b->C, b->N, b->C32);
#endif
}

static void bench_track(const BenchBuf *b) {
    float mx = 0.0f;
#if defined(__aarch64__) || defined(__arm64__) || defined(_M_ARM64)
    if (fe_qgemm_ops.tier == FE_QGEMM_TIER_ARM_I8MM)
        qgemm_i8mm_fp32_fused_track_maxabs(b->M, b->N, b->K, b->Aq, b->Bp, b->combined, b->bias, b->C, b->N, b->C32, &mx);
    else if (fe_qgemm_ops.tier == FE_QGEMM_TIER_ARM_DOTPROD)
        qgemm_dotprod_fp32_fused_track_maxabs(b->M, b->N, b->K, b->Aq, b->Bp, b->combined, b->bias, b->C, b->N, b->C32, &mx);
    else
        qgemm_neon_fp32_fused_track_maxabs(b->M, b->N, b->K, b->Aq, b->Bp, b->combined, b->bias, b->C, b->N, b->C32, &mx);
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    if (fe_qgemm_ops.tier == FE_QGEMM_TIER_X86_AVX512_VNNI)
        qgemm_avx512vnni_fp32_fused_track_maxabs(b->M, b->N, b->K, b->Aq, b->Bp, b->combined, b->bias, b->C, b->N, b->C32, &mx);
    else if (fe_qgemm_ops.tier == FE_QGEMM_TIER_X86_AVX_VNNI)
        qgemm_avxvnni_fp32_fused_track_maxabs(b->M, b->N, b->K, b->Aq, b->Bp, b->combined, b->bias, b->C, b->N, b->C32, &mx);
    else
        qgemm_avx2_fp32_fused_track_maxabs(b->M, b->N, b->K, b->Aq, b->Bp, b->combined, b->bias, b->C, b->N, b->C32, &mx);
#endif
    g_sink += mx;
}

static void bench_gru_kernel(const BenchBuf *b) {
    const int M = FE_F2;
    const int D = FE_C2;
    uint16_t *h_fp16 = b->A_fp16;
    float *h_out = b->C;
    const float *br = b->bias;
    const float *bz = b->bias + D;
    const float *bn_i = b->bias + 2 * D;
    const float *bn_h = b->bias;
#if defined(__aarch64__) || defined(__arm64__) || defined(_M_ARM64)
    if (fe_qgemm_ops.tier == FE_QGEMM_TIER_ARM_I8MM)
        qgemm_i8mm_gru_full_fused_fp16inout(M, D, b->Aq, D, b->Aq, D, b->Bp, b->Bp,
                                            b->combined, b->bias, b->combined, b->bias,
                                            br, bz, bn_i, bn_h, h_fp16, D, h_out, D);
    else if (fe_qgemm_ops.tier == FE_QGEMM_TIER_ARM_DOTPROD)
        qgemm_dotprod_gru_full_fused_fp16inout(M, D, b->Aq, D, b->Aq, D, b->Bp, b->Bp,
                                               b->combined, b->bias, b->combined, b->bias,
                                               br, bz, bn_i, bn_h, h_fp16, D, h_out, D);
    else
        qgemm_neon_gru_full_fused_fp16inout(M, D, b->Aq, D, b->Aq, D, b->Bp, b->Bp,
                                            b->combined, b->bias, b->combined, b->bias,
                                            br, bz, bn_i, bn_h, h_fp16, D, h_out, D);
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    if (fe_qgemm_ops.tier == FE_QGEMM_TIER_X86_AVX512_VNNI)
        qgemm_avx512vnni_gru_full_fused_fp16inout(M, D, b->Aq, D, b->Aq, D, b->Bp, b->Bp,
                                                  b->combined, b->bias, b->combined, b->bias,
                                                  br, bz, bn_i, bn_h, h_fp16, D, h_out, D);
    else if (fe_qgemm_ops.tier == FE_QGEMM_TIER_X86_AVX_VNNI)
        qgemm_avxvnni_gru_full_fused_fp16inout(M, D, b->Aq, D, b->Aq, D, b->Bp, b->Bp,
                                               b->combined, b->bias, b->combined, b->bias,
                                               br, bz, bn_i, bn_h, h_fp16, D, h_out, D);
    else
        qgemm_avx2_gru_full_fused_fp16inout(M, D, b->Aq, D, b->Aq, D, b->Bp, b->Bp,
                                            b->combined, b->bias, b->combined, b->bias,
                                            br, bz, bn_i, bn_h, h_fp16, D, h_out, D);
#endif
}

typedef void (*bench_fn)(const BenchBuf *);

static void op_raw_int32(const BenchBuf *b) {
    fe_qgemm_ops.gemm_int32(b->M, b->N, b->K, b->Aq, b->Bp, b->C32, b->N);
}

static void op_raw_k20(const BenchBuf *b) { bench_int32_k20(b); }

static void op_fp32_fused(const BenchBuf *b) {
    fe_qgemm_ops.gemm_fp32_fused(b->M, b->N, b->K, b->Aq, b->Bp,
                                   b->combined, b->bias, b->C, b->N, 0, b->C32);
}

static void op_fp32_silu(const BenchBuf *b) {
    fe_qgemm_ops.gemm_fp32_fused(b->M, b->N, b->K, b->Aq, b->Bp,
                                   b->combined, b->bias, b->C, b->N, 1, b->C32);
}

static void op_fused_acc(const BenchBuf *b) { bench_acc(b); }
static void op_track_maxabs(const BenchBuf *b) { bench_track(b); }

static void op_calib_transposed(const BenchBuf *b) {
    FeActScale act; fe_actscale_init(&act);
    fe_qgemm_packed_calib_transposed_in(b->M, b->N, b->K, b->A_KM,
                                          b->Bp, b->scales_w, b->row_sums,
                                          b->bias, b->C, b->N,
                                          b->aq, b->C32, &act);
}

static void op_calib_transposed_fp16(const BenchBuf *b) {
    FeActScale act; fe_actscale_init(&act);
    fe_qgemm_packed_calib_transposed_in_fp16(b->M, b->N, b->K, b->A_fp16,
                                               b->Bp, b->scales_w, b->row_sums,
                                               b->bias, b->C, b->N,
                                               b->aq, b->C32, &act);
}

static void op_concat2_fp16b(const BenchBuf *b) {
    FeActScale act; fe_actscale_init(&act);
    fe_qgemm_packed_silu_calib_concat2_fp16b(b->M, b->N, b->K / 2,
                                               b->A, b->B_fp16,
                                               b->Bp, b->scales_w, b->row_sums,
                                               b->bias, b->C, b->N,
                                               b->aq, b->C32, &act);
}

static void op_calib_acc(const BenchBuf *b) {
    FeActScale act; fe_actscale_init(&act);
    fe_qgemm_packed_calib_acc(b->M, b->N, b->K, b->A,
                                b->Bp, b->scales_w, b->row_sums,
                                b->bias, b->C, b->N,
                                b->aq, b->C32, b->fp32_scratch, &act);
}

static void op_to_int8out(const BenchBuf *b) {
    FeActScale in_act, out_act;
    fe_actscale_init(&in_act);
    fe_actscale_init(&out_act);
    (void)fe_qgemm_packed_calib_to_int8out(b->M, b->N, b->K, b->A,
                                             b->Bp, b->scales_w, b->row_sums,
                                             b->bias, b->Cq, b->N,
                                             b->C2, b->N,
                                             b->aq, b->C32, &in_act, &out_act);
}

static void op_gru_full(const BenchBuf *b) { bench_gru_kernel(b); }

static BenchResult run_bench(const char *family, const char *op,
                             const BenchShape *shape, bench_fn fn,
                             int iters, int repeats, int checksum_kind) {
    BenchBuf b;
    memset(&b, 0, sizeof(b));
    fill_case(&b, shape->M, shape->N, shape->K);

    for (int i = 0; i < 8; ++i) fn(&b);

    double samples[32];
    if (repeats > 32) repeats = 32;
    for (int r = 0; r < repeats; ++r) {
        double t0 = now_s();
        for (int i = 0; i < iters; ++i) fn(&b);
        double t1 = now_s();
        samples[r] = (t1 - t0) * 1.0e6 / (double)iters;
    }

    double sum;
    if (checksum_kind == 0) sum = checksum_i32(b.C32, shape->M * shape->N);
    else if (checksum_kind == 1) sum = checksum_i8(b.Cq, shape->M * shape->N);
    else sum = checksum_f32(b.C, shape->M * shape->N);
    g_sink += sum;

    BenchResult res;
    res.family = family;
    res.op = op;
    res.shape = shape;
    res.iters = iters;
    res.repeats = repeats;
    res.median_us = median(samples, repeats);
    res.gmac_s = shape->macs / (res.median_us * 1.0e-6) / 1.0e9;
    res.checksum = sum;
    free_case(&b);
    return res;
}

static void print_result(const char *tier, const BenchResult *r) {
    printf("%s,%s,%s,%d,%d,%d,%.0f,%d,%d,%.3f,%.3f,%.6f\n",
           tier, r->family, r->op,
           r->shape->M, r->shape->N, r->shape->K, r->shape->macs,
           r->iters, r->repeats, r->median_us, r->gmac_s, r->checksum);
}

static int run_tier(const char *tier, int iters, int repeats) {
    if (fe_qgemm_force_tier(tier) != 0) return 0;
    if (fe_qgemm_init() != 0) return 0;

    const BenchShape winograd = {"winograd_inner", FE_F1 / 2, FE_C1, FE_C1,
                                 (double)(FE_F1 / 2) * FE_C1 * FE_C1};
    const BenchShape rf_fc = {"rf_fc", FE_F2, FE_C2, FE_C2,
                              (double)FE_F2 * FE_C2 * FE_C2};
    const BenchShape qkv = {"qkv_proj", FE_F2, 3 * FE_C2, FE_C2,
                            (double)FE_F2 * (3 * FE_C2) * FE_C2};
    const BenchShape dec = {"dec_concat1x1", FE_F1, FE_C1, 2 * FE_C1,
                            (double)FE_F1 * FE_C1 * (2 * FE_C1)};
    const BenchShape k20 = {"attn_qk_k20", FE_F2, FE_F2, FE_HEAD_DIM_PAD_K,
                            (double)FE_F2 * FE_F2 * FE_HEAD_DIM_PAD_K};
    const BenchShape gru = {"gru_full_fused", FE_F2, 3 * FE_C2, FE_C2,
                            2.0 * FE_F2 * (3 * FE_C2) * FE_C2};

    BenchResult r;
    r = run_bench("kernel", "raw_int32", &winograd, op_raw_int32, iters, repeats, 0);
    print_result(tier, &r);
    r = run_bench("kernel", "fp32_fused", &winograd, op_fp32_fused, iters, repeats, 2);
    print_result(tier, &r);
    r = run_bench("kernel", "fp32_fused_silu", &winograd, op_fp32_silu, iters, repeats, 2);
    print_result(tier, &r);
    r = run_bench("kernel", "fused_acc", &rf_fc, op_fused_acc, iters, repeats, 2);
    print_result(tier, &r);
    r = run_bench("kernel", "track_maxabs", &qkv, op_track_maxabs, iters, repeats, 2);
    print_result(tier, &r);
    r = run_bench("kernel", "raw_int32_k20", &k20, op_raw_k20, iters, repeats, 0);
    print_result(tier, &r);
    r = run_bench("kernel", "gru_full_fused_fp16", &gru, op_gru_full, iters, repeats, 2);
    print_result(tier, &r);

    r = run_bench("wrapper", "calib_transposed", &rf_fc, op_calib_transposed, iters, repeats, 2);
    print_result(tier, &r);
    r = run_bench("wrapper", "calib_transposed_fp16", &rf_fc, op_calib_transposed_fp16, iters, repeats, 2);
    print_result(tier, &r);
    r = run_bench("wrapper", "concat2_fp16b", &dec, op_concat2_fp16b, iters, repeats, 2);
    print_result(tier, &r);
    r = run_bench("wrapper", "calib_acc", &rf_fc, op_calib_acc, iters, repeats, 2);
    print_result(tier, &r);
    r = run_bench("wrapper", "to_int8out", &qkv, op_to_int8out, iters, repeats, 1);
    print_result(tier, &r);

    return 1;
}

int main(int argc, char **argv) {
    const char *tier_arg = argc > 1 ? argv[1] : "all";
    int iters = argc > 2 ? atoi(argv[2]) : 2000;
    int repeats = argc > 3 ? atoi(argv[3]) : 7;
    if (iters <= 0) iters = 2000;
    if (repeats <= 0) repeats = 7;

    printf("tier,family,op,M,N,K,macs,iters,repeats,median_us,gmac_s,checksum\n");
    if (strcmp(tier_arg, "all") != 0) {
        if (!run_tier(tier_arg, iters, repeats)) {
            fprintf(stderr, "tier '%s' not available\n", tier_arg);
            return 2;
        }
        fprintf(stderr, "sink=%f\n", g_sink);
        return 0;
    }

    const char *tiers[] = {
#if defined(__aarch64__) || defined(__arm64__) || defined(_M_ARM64)
        "i8mm", "dotprod", "neon"
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
        "avx512vnni", "avxvnni", "avx2"
#endif
    };
    int ran = 0;
    for (size_t i = 0; i < sizeof(tiers) / sizeof(tiers[0]); ++i)
        ran += run_tier(tiers[i], iters, repeats);
    fprintf(stderr, "sink=%f\n", g_sink);
    return ran ? 0 : 2;
}
