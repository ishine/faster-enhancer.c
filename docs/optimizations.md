# Optimization design notes

<sub>W8A8 quantization · 6-tier int8 GEMM microkernels · fp16 cross-stage storage · op fusion · measurement methodology.<br>
The model's own spec (shapes · equations · layer inventory) is in <a href="architecture.md">architecture.md</a>; the canonical RTF/quality tables are in the root <a href="../README.md">README</a>.</sub>

> **TL;DR** — the dominant design constraint is not performance but **cross-tier byte-identity**. The universal `[−127,127]` clamp, forcing FMA3 (dropping SSE4.1), and reproducing the 12-bit `VRCPPS` on AVX-512 all follow from it. On top of that constraint each tier picks the strongest int8 dot instruction its ISA offers and reaches **85–93 % of ISA peak**.

---

## 1. W8A8 quantization

| Item | Design |
|---|---|
| **Weight** | per-output-row **symmetric int8**, `scale = max\|w\|/127`, zero-point 0 |
| **Activation** | per-tensor **asymmetric u8**, **recomputed every frame** (ORT `DynamicQuantizeLinear` semantics) |
| **Storage** | activations stored as `int8 = u8 − 128` → reuse the signed×signed kernel with no XOR/shift |
| **Clamp** | universal `[−127, 127]` on both weights and activations |
| **Bias correction** | zp offset absorbed into the fp32 term of the dequant epilogue:<br>`bias_eff = bias + scale_x · scale_w[n] · (128 − zp_x) · row_sums[n]` |
| **Calibration freeze** | none. `FeActScale` only records running min/max for diagnostics |

Dequant is `y[m,n] = scale_x · scale_w[n] · (C32[m,n] + (128 − zp_x)·row_sums[n]) + bias[n]`; in practice it is pre-folded into `combined_scale[n]` / `bias_eff[n]` and handled by a single FMA in the kernel epilogue.

### Why these choices

| Decision | Rationale |
|---|---|
| **dynamic** activations (not static) | freezing a static calibration regresses into phoneme blur. The per-frame min/max scan is a small fixed cost |
| **asymmetric u8** activations | recovers 2–16 kHz consonant information versus symmetric. The zp offset is absorbed into the bias, so the dot cost is 0 |
| **per-row symmetric** weights | preserves per-row dynamic range. zp=0, so no weight-side row_sums correction is needed |
| **W8** (no sub-byte) | W7 loses PESQ −0.12 to saturation accumulation in the AVX2 `vpmaddubsw` polyfill |

Most of the model's weights are load-bearing — even 10 % 0-shot magnitude pruning fails it, PESQ 1.66 → 1.19. W8A8 is close to the lowest bit width that preserves quality.

### The universal `[−127, 127]` clamp

The int8 range is `[−128, 127]`, but `−128` is thrown away. That one line buys three things at once.

**(a) saturation-free dot** — overflow is mathematically impossible even on the tightest i16 accumulation path.

| path | worst-case | limit | margin |
|---|---|---|---|
| AVX2 `vpmaddubsw` pair-sum | 127×127 + 127×127 = 32,258 | INT16_MAX 32,767 | 509 |
| NEON 2-step `vmlal_s8` chain | 2 × 127×127 = 32,258 | INT16_MAX 32,767 | 509 |

This is also why the NEON i16 chain is limited to exactly 2 K-steps — a third `vmlal_s8` overflows.

**(b) cross-tier byte-identity** — AVX2 has no native int8 dot, so it moves the sign with `_mm256_sign_epi8`, and that instruction cannot negate `−128` (`−(−128) = +128` is not representable in int8). At `a = −128` it drifts 1 LSB from tiers that use a native dot, breaking byte-id. Clamping `−128` away removes the edge case entirely.

**(c)** The price is ~1 LSB of precision in the most-negative bin, and **PESQ Δ = 0.000** (3 dp).

> The clamp is not a single-tier change, it is the **budget basis for every downstream quant trick**.

### Quality

3 clips (babble/car/street @5 dB) · 5569-frame window · Apple M2 / I8MM · re-measured 2026-07-25. The ARM family is byte-id, so all three tiers share the same output quality.

**absolute quality against the clean target**

| variant | PESQ-WB | STOI | eSTOI | SNR dB |
|---|---:|---:|---:|---:|
| `noisy_5dB` | 1.120 | 0.8576 | 0.6607 | 4.98 |
| `onnx_fp32` | 1.833 | 0.9440 | 0.8721 | 15.07 |
| **`fe_q8`** | **1.837** | 0.9442 | 0.8726 | 15.01 |

**faster-enhancer.c q8 vs fp32 ONNX fidelity**

| clip | PESQ-WB | STOI | eSTOI | SNR dB | max\|d\| |
|---|---:|---:|---:|---:|---:|
| babble_5dB | 4.569 | 0.9991 | 0.9981 | 32.47 | 0.0357 |
| car_5dB | 4.595 | 0.9998 | 0.9996 | 36.06 | 0.0117 |
| street_5dB | 4.573 | 0.9991 | 0.9984 | 33.60 | 0.0399 |
| **mean** | **4.579** | 0.9993 | 0.9987 | **34.04** | 0.0291 |

<details>
<summary><b>the reference must sit on the same causal STFT grid as the engine</b> — measurement methodology correction (2026-07-25)</summary>

<br>

The earlier reference (center=True, since deleted) called `torch.stft` without a `center=` argument → PyTorch's default `center=True`, i.e. frames are **centered** on `t·hop` after `n_fft/2 = 512` reflect padding. The C engine, by contrast, is a causal streaming STFT where frame t covers `[t·hop − 704, t·hop + 320)` (`center=False` + 704-sample zero left-history).

The two analysis grids are offset by `704 − 512 = 192` samples, and 192 is not a multiple of hop 320, so it does not cancel. Running the **same fp32 model** on the two grids 192 samples apart and comparing already yields SNR ≈ 22.8 dB · PESQ-WB ≈ 4.30 — indistinguishable from what used to be reported as "q8 vs fp32 fidelity" (4.253 / 24.73 dB). That table was measuring a **framing phase offset**, not int8 quantization error.

`tools/onnx_reference_causal.py` replicates the framing and synthesis-window normalization of `src/fe_stft.c` exactly. Once aligned, the q8 output lines up with the reference at **lag 0** (704 was needed before), and fidelity rises from PESQ 4.253 → **4.579** and SNR 24.73 → **34.04 dB**. The correction happens to run in the project's favor — the earlier numbers were understating actual quantization fidelity.

The absolute-quality table is barely affected by the correction: causal-grid fp32 ONNX is PESQ 1.833 (previously 1.835 non-causal), i.e. the 512-sample look-ahead advantage of `center=True` is negligible on this task.

</details>

---

## 2. GEMM microkernels (6 tiers)

Every tier implements the same fused-kernel family. There is no scalar fallback — `fe_qgemm_init` checks the baseline ISA and, if it is not met, `fe_init` returns non-zero (no silent slow path).

| tier | strongest int8 dot | MAC/inst | tile | ISA-peak | target |
|---|---|---:|---|---:|---|
| **NEON** (floor) | `vmull_s8`→`vmlal_s8`→`vpadalq_s16` | 8 | 4×8 | ~88 % | arm64 baseline |
| **DOTPROD** | `vdotq_laneq_s32` | 16 | 8×8 | ~85–87 % | arm64 + FEAT_DotProd |
| **I8MM** | `vmmlaq_s32` (2×8·8×2→2×2) | 32 | 8×8 | ~87 % | arm64 + FEAT_I8MM |
| **AVX2** (x86 floor) | `vpmovsxbw` + `vpmaddwd` | ~24 | 12×8 | ~90 % | Haswell+ (AVX2+FMA3+F16C) |
| **AVX-VNNI** | `vpdpbusd ymm` | 32 | 12×8 | ~93 % | Alder Lake+, Zen 4 |
| **AVX-512-VNNI** | `vpdpbusd zmm` | 64 | 12×16 | zmm-full | Ice Lake / SPR, Zen 4 |

The kernel set every tier implements: `int32` / `int32_k20`† / `fp32_fused` / `fp32_fused_acc` / `fp32_fused_track_maxabs` / `gru_full_fused_fp16inout`.

† `_int32_k20` (full unroll for the MHSA Q@Kᵀ, K=20) is **unimplemented on I8MM only**. K=20 is not a multiple of 8 and does not divide into SMMLA k8-groups, so even on I8MM hardware the attention score GEMM routes to `qgemm_dotprod_int32_k20` every frame — this is a **steady-state path**, not a remainder fallback.

### Key technique per tier

**NEON** — ARMv8.0 has no int8 dot. Start with `vmull_s8`, chain two K-steps into an i16 accumulator with `vmlal_s8`, then promote to i32 with a single `vpadalq_s16` (the ggml q3_K pattern). The core of the 4×8 microkernel is loading A's 4-byte K-quad **once per row** with `vld1_dup_s32` and reusing it across all of cols 0-7 — half the A-side LD1R traffic of the previous 8×4-twice path. The register-resident peak of `vmull+vpadalq ~55 GMAC/s` is structurally low.

**DOTPROD** — the bottleneck was not the dot instruction but the **A-operand load**. The legacy path rebuilt SDOT's `a` with `vld1q_dup_s32` (LD1R) for every row and every k4-group, throttling to ~62 % of SDOT peak (136 GMAC/s) on M2. Fixed by **A row-quad pre-pack + lane-indexed dot**: repack A once so each k4-group holds the 4-byte quads of 8 rows contiguously, then load just two `vld1q_s8` and feed them into lanes 0..3 of `vdotq_laneq_s32` — all LD1R gone, ~186 GMAC/s (85–87 % of peak).

**I8MM** — SMMLA's `a` operand is 16 B = `[r0_K0..7 | r1_K0..7]`. The legacy path assembled it every K-block with 8 `vld1_s8` + 4 `vcombine` = 12 µops. Building the same layout ahead of time with an **A row-pair pre-pack** (KleidiAI-style) lets a single `vld1q_s8_x4` µop cover all 4 row-pairs — 8 µops saved per K-block. The epilogue gathers the 2×2 SMMLA outputs into row-major order with `vtrn1q/vtrn2q_s64`. ~183 GMAC/s.

**AVX2** — the classic `vpmaddubsw` is unusable because it can saturate (u8×i8 pair-sum max `2×255×127 = 64,770 > 32,767`). The current path is `vpmovsxbw` + `vpmaddwd` (XNNPACK-style): i16×i16 → i32 pair-sum, the only bit-id-safe path where saturation is impossible by construction. B is re-packed K-pair-interleaved once per N-block, A is pre-expanded to i16 once per call. **ISA floor**: 8 `vpmaddwd` on p01 per K-quartet = 8 cyc, exactly 2× AVX-VNNI (4 cyc) — that is the intrinsic limit of bit-id-safe AVX2.

**AVX-VNNI** — `vpdpbusd` wants (u8, i8), so all of A gets `XOR 0x80` 1× per call (`prexor128_a`), the inner loop does only broadcast + vpdpbusd, and the correction term `wsum[n] = 128·Σ_K b[n,k]` is subtracted at the end. Three techniques:

- **memory-source broadcast** — `_mm256_set1_epi32(*(int32_t*)p)` lowers to `vmovd; vpbroadcastd ymm,xmm`, occupying **p5 only**, and vpdpbusd is p0+p5, so it steals half the dot slots. Forcing `_mm256_broadcastd_epi32(_mm_loadu_si32(p))` emits one `vpbroadcastd ymm,[mem]` µop (load port).
- **12-row tile** — vpdpbusd is lat 5 / tput 2. 8 chains run at 8/5 = 1.6/cyc, latency-bound (80 % of peak); 12 chains saturate dispatch (~93 %). Register budget 14 ymm, no spills.
- **wsum 4-way chain-split** — with a single accumulator the chain depth is k4_groups·5 (K=72 → 90 cyc vs 18 cyc of dispatch). Splitting 4 ways gives depth ~25 cyc. Integer addition is associative and saturation-free, so bit-id holds.

The M-tail must stay on SIMD too — the shared `qgemm_scalar_int32` reads the prexored u8 A as signed and applies no wsum correction, so it produces **wrong values**. Winograd M=64 runs the 12-loop over `(M/24)*24` so the remainder is 8-aligned (`4×12 + 2×8`).

**AVX-512-VNNI** — the whole tier is 16-lane zmm. The per-frame activation quantize is also zmm-ized via a `target("avx512f,avx512bw,avx512vl")` clone (runtime dispatch through the `fe_qg_x86_avx512` flag) — this removed a latent perf bug where an 8-wide quantize was feeding a 16-wide GEMM. The N%16==8 leftover goes through the ymm helper rather than masked zmm (avoids wasting lanes).

> **cross-tier bit-id bug (found with Intel SDE)** — the avx512vnni output was **not** byte-identical to avx2/avxvnni, differing at SNR 44 dB. The cause was reciprocal precision in sigmoid/tanh: ymm uses `_mm256_rcp_ps` (~12-bit), zmm uses `_mm512_rcp14_ps` (14-bit) → divergence accumulating through the GRU gates / SiLU. `fe_avx512_rcp_match256` splits the zmm denominator into two ymm halves and reproduces the 12-bit reciprocal bit-for-bit with `_mm256_rcp_ps` (AVX-512F only).

### ISA floor / ceiling

The dense-equivalent runtime multiply footprint of the Winograd k=3 path is **46.3 MMAC/frame** (direct-conv equivalent 53.4). Against a 6.667 ms budget, x86 at ~3.8 GHz:

| tier | peak MAC/cyc | GEMM-only floor | realistic floor +overhead |
|---|---:|---:|---:|
| AVX-VNNI | 64 | ~0.029 | **~0.04** |
| AVX2 | ~24 | ~0.076 | **~0.09** |
| AVX-512-VNNI | 128 | ~0.014 | **~0.03** |

AVX2 0.07 is below the floor, and the path to 0.03 is the AVX-512 zmm tier, not AVX-VNNI (ymm).

**Apple M2 ISA ceiling** — measured register-resident peak is SMMLA **211 GMAC/s**, SDOT **220**. `vdotq_s32` is 4 pipes × 16 = 64 MAC/cyc and `vmmlaq_s32` is 2 pipes × 32 = 64 — I8MM's 2× MAC density is **exactly cancelled** by having half the dispatch ports. Hence DOTPROD GEMM (186) ≈ I8MM (183) on M2. There is no further win in the SIMD domain on M2; what is left is other ARM cores (4-pipe vmmlaq) or the algorithm level.

Register spills in the hot tile of every tier are 0 — intrinsics + target clones give the compiler freedom over register allocation and scheduling, so hand-written asm buys nothing and only adds per-uarch maintenance.

**BSS pre-fault** — the 363,520-byte scratch of the AVX2 i16-expand path drove the peak/p50 ratio to 5.78× through first-touch page faults. `fe_qgemm_init` now touches every page once (`qgemm_avx2_prefault_buffers`), moving it off the normal path — peak −66 %, ratio 2.1×. Both ARM tiers do the same (`a_rp` plus the GRU-specific `gru_x_rp` / `gru_h_rp`).

---

## 3. fp16 cross-stage storage

Long-lived state between layers is stored as IEEE binary16. Exactly two buffers qualify — `enc_skip` (encoder→decoder skip) and `gru_h` (recurrent state). Intra-frame scratch such as `attn_*` is excluded because it is zero-re-initialized on every call.

| buffer | fp32 | fp16 | saving |
|---|---:|---:|---:|
| `enc_skip[4]` | 196,608 B | 98,304 B | −98,304 B |
| `gru_h[4]` | 82,944 B | 41,472 B | −41,472 B |
| `enc_skip_scratch` (staging, removed) | 49,152 B | 0 B | −49,152 B |
| **Named state/staging delta** | | | **−188,928 B (184.5 KiB)** |

Currently `sizeof(FeState) = 432,384` bytes.

int8 storage was rejected — the joint min/max round-trip in `dec_1x1` lost ~40 dB of SNR. fp16 keeps 10 mantissa bits.

**GRU dual-store ngate epilogue** — the kernel reads the previous hidden state directly from the fp16 slot (`fe_fp16_load4`) and writes the updated state **simultaneously** as fp32 (for the immediate consumer `rnn_fc`) and fp16 (for the next frame's `gru_h`). This removes both the unpack pass before the GRU and the pack pass after it that the old fp32-inout path required.

**fp16 conversion is mandatory** — F16C (`VCVTPS2PH`/`VCVTPH2PS`) on x86, `FCVTN`/`FCVTL` on aarch64. There is no scalar fallback in the build; an unsupported host is handled as an `fe_init` failure (avoiding the risk of silently not running).

**Page hygiene** — `fe_init` best-effort page-locks the blob, the weight buffers and `FeState` (`mlock`/`VirtualLock`) **immediately before** warmup. The lock goes before warmup so that warmup does not itself generate the soft faults it was meant to avoid. A 128-frame silent warmup then heats the weight pages, the BSS scratch, the branch predictor and L1/L2. Activation scales are not frozen during warmup.

**Working-set** — the Winograd runtime scratch is `wino_U` (98,304 B) + `wino_dq` (24,576 B) = 122,880 B, treated as L2-resident. The 258,048 B of weight-derivation scratch is `malloc`/`free`d at init time and is not resident.

**Build-flag hygiene** — on x86, besides the per-tier per-file `-march`, a list of 10 TUs (`FE_X86_AVX2_TUS`) gets `-mavx2 -mfma -mf16c` attached separately. A TU missing from that list has its SIMD blocks **silently disappear** in preprocessing — a latent bug that actually happened in `fe_winograd.c`.

---

## 4. Op fusion

Adjacent ops are merged to remove fp32 intermediate-buffer round-trips, separate transposes and separate packs. Stage boundaries such as *dequant epilogue → input quantize of the next op* are the prime target.

| fusion | what |
|---|---|
| **transpose + quantize** | the `[F,C] → [C,F]` transpose of the freq-axis Linear is folded into the int8 quantize pass (4×4 fp32 tile transpose + saturating narrow) |
| **residual in epilogue** | `resA`/`resB` absorbed into the accumulating store (`C += …`) of the `rnn_fc`/`attn_fc` dequant epilogue instead of a separate add pass |
| **bias_eff** | zp offset correction pre-folded into the bias, dot cost 0 |
| **GRU full-fusion** | `W_ih·x` + `W_hh·h` + r/z/n gates + h-update in one kernel per tier. The int32 accumulators never spill to memory and land in the L1d-resident `ih_tile` |
| **Winograd batched epilogue** | 4× int32 U → dequant + Aᵀ + bias + optional SiLU + fp16 skip write in a single pass |
| **softmax-from-int32** | takes int32 scores directly and does dequant→softmax→int8 in 2 passes. Uses only row scratch, never materializes the full fp32 score matrix |
| **max-abs tracking** | the max\|y\| for the next layer's requantize scale is tracked in a register during the dequant store instead of a separate scan |
| **concat + quantize** | the decoder 1×1 merges the min/max of fp32 A and fp16 B into one (scale, zp) and emits directly into the same aq row |
| **STFT window pre-fold** | the 1/N of the iFFT is pre-folded into the synthesis window and removed from the kernel |

**Winograd F(2,3)** — the k=3 convs of the encoder/decoder (Cin=Cout=96) run as Winograd, cutting multiplies to **2/3** (direct `3·Ci·Co·F` → `4·(F/2)·Co·Ci`). The input transform is **two-pass**: pass A computes Bᵀ purely to extract the per-component max-abs of the 4 components and throws the result away, then pass B recomputes Bᵀ and quantizes to int8 straight away with the now-fixed scale — the call is that recomputation is cheaper than the memory traffic of keeping the 4 fp32 transform results (98 KiB) in L1d. The transform matrices contain only ±1 and ±0.5, so the runtime transform is add/sub only (0.5 appears only in the offline weight transform). The per-tile fused microkernel was prototyped and then dropped — at MR=2 it loses weight reuse and was slower than the batched path on every tier measured.

**Limits of fusion** — the production k=3 path is Winograd-only; there is no direct/im2col k=3 path. The only real im2col is the strided `enc_pre` conv, and there `K = 2·Ci = 16` is too small for int8 to amortize, so it was left as fp32 sgemm.

---

## 5. Measurement methodology

### byte-identity regression gate

For the same input, all SIMD tiers **within an architecture family** produce byte-identical output (same SHA256): x86 `AVX2 ≡ AVX-VNNI ≡ AVX-512 VNNI`, ARM `NEON ≡ DOTPROD ≡ I8MM`.

The preconditions are: no `-ffast-math`, identical coefficients and identical reduction order within the family, and the universal clamp. **Cross-architecture** (x86 ↔ ARM) is not byte-id because of IEEE round-tie behavior differences, and is asserted through STOI/SNR equivalence instead.

| method | scope | what |
|---|---|---|
| `tools/sha256_matrix.sh` | every runnable tier | WAV hashes for 3 clips × n tiers. Rows within a family matching = PASS. An empty pre/post diff means no regression |
| `tests/fe_test_cross_tier.c` | every built tier | cross-tier SNR diagnostics + final float bit-id memcmp |
| Intel SDE (`-spr`) | AVX-512 VNNI | the dev box has AVX-512 fused off, so functional verification only. No timing |

> **Regression workflow** — run `sha256_matrix.sh` before and after any non-trivial code change and `diff`. Empty diff = no byte-level regression.

SSE4.1 was permanently dropped: without FMA3 the dequant epilogue becomes a 2-step `mul → add` rounding → ~1 ULP/op drift → byte-id collapse once it accumulates. The x86 floor tier is named AVX2, but the runtime requirement is **AVX2+FMA3+F16C**.

### Quality measurement

The fp32 ONNX reference is generated on the **same causal STFT grid as the engine** (`tools/onnx_reference_causal.py`) — see the folded section under §1 Quality for why. `tools/quality_compare.py` estimates the integer lag by cross-correlation, aligns, then computes PESQ-WB / STOI / eSTOI / SNR. When the grids match, the lag is 0.

### hybrid-CPU latency tail

On an i5-13400F (6 P + 4 E) at default priority, p50→p95 is tight (~1.2×) but p99→peak blows up, and the single peak frame **swings wildly run to run** (VNNI max 0.89 ms → 3.94 ms on the next run). That instability is the tell — a deterministic effect would be stable.

Pinning isolates the cause (babble 5569 frames, VNNI, **ms/frame**):

| pin | p50 | p95 | p99 | peak | note |
|---|---:|---:|---:|---:|---|
| **single P-core** (LP4) | 0.403 | 0.416 | 0.437 | 0.95 | **p99 = 1.08× p50** — flat |
| **single E-core** (LP12) | **1.435** | 1.495 | 1.538 | 2.96 | **3.6× slower p50** |
| all P-core, normal prio | 0.382 | 0.496 | 0.642 | 2.17 | avoids E, still preempted |
| unpinned, normal prio | 0.385 | 0.506 | 0.697 | 3.96 | baseline tail |

Two facts drop out. (1) On a P-core the compute is intrinsically jitter-free — there is no separable kernel-side tail. (2) The Gracemont E-core has 1 256-bit VNNI port (Raptor Cove has 2), so it runs the same frame 3.6× slower. *Every* frame the scheduler parks on an E-core becomes a p95–p99 tail sample.

**The lever is thread priority, not affinity:**

| config | p50 | p99 | peak |
|---|---:|---:|---:|
| normal priority | 0.378 | 0.633 | 1.4–5.2 |
| **`THREAD_PRIORITY_TIME_CRITICAL`** | 0.40 | **0.50** | **0.6–1.0** |
| single-P-core pin, *normal* prio | 0.40 | 0.918 | 6.45 ← worse |

The Win11 Thread Director steers high-priority threads onto P-cores on its own, so E-core avoidance comes free with the priority bump alone. Conversely, single-core pinning at normal priority is *worse* — if the OS schedules anything on that core there is no P-core left to fall back to.

**Why this lives outside the library** — thread scheduling belongs to the integrator, and a library that quietly raises its caller's priority violates least-surprise. The engine is byte-identical regardless of priority; only the jitter envelope changes. The only OS hygiene the library does is page-locking and warmup, and both are guarded and no-op on failure.

```c
// Windows (the strongest guarantee is MMCSS AvSetMmThreadCharacteristics L"Pro Audio")
SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);
// macOS — QoS keeps the thread on a P-core, no privileges needed
pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
// Linux — SCHED_FIFO if RT-capable, otherwise a harmless no-op
struct sched_param sp = { .sched_priority = 50 };
pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp);
```

`fe_run_file` applies the platform-appropriate call by default; `FE_RT=0` lets you measure the normal-priority envelope for comparison.

---

## Provenance of the numbers

The numbers in this document come in two kinds. They are separated here so that anyone who clones the public repo to check them is not confused.

**Numbers with artifacts under `experiments/`** — runner logs, summaries and raw CSVs are committed, together with the reproduction scripts:
M2 tier/system/Winograd/build-option ablations, qgemm profiles, benchmarks against ONNX,
VoiceBank-DEMAND 824-utterance quality, the M2 energy matrix, M2 30-minute paced sustained,
S23+ tier/placement/sustained.

**Numbers measured during development for which no artifact was kept** — the figures below exist only as prose in this document.
They are not to be cited in the paper until re-measured:

| number | what |
|---|---|
| PESQ 1.66 → 1.19 | 10 % 0-shot magnitude pruning. No experiment script, no log |
| W7 −0.12 PESQ | AVX2 `vpmaddubsw` polyfill saturation. The measurement script was deleted along with it |
| hybrid-CPU latency tail table | i5-13400F LP4/LP12 pinning. There is no x86 directory under `experiments/` at all |
| BSS pre-fault peak/p50 5.78× → 2.1× | page-fault jitter measurement. No log |

The x86 rows of Table 1 (AVX-VNNI 0.057 / AVX2 0.110) are in the same state — measured before
the current repeat protocol, with no artifact. The paper's limitations section says so.

## Sources

| source | technique |
|---|---|
| [ggml q3_K / tq1_0](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cpu/arch/arm/quants.c) | NEON `vmull→vmlal_s8 chain→vpadalq_s16` i16-level fusion |
| ggml / llama.cpp Q8_0 | AVX2 vpsignb saturation-free dot (now replaced by the vpmaddwd i16-expand) |
| [XNNPACK](https://github.com/google/XNNPACK) `qd8-f32-qc8w-gemm-4x8c8-avx2` | AVX2 `vpmovsxbw + vpmaddwd` 3-op dot |
| ARM KleidiAI | I8MM A row-pair pre-pack layout |
| [insn_bench_aarch64](https://github.com/ocxtal/insn_bench_aarch64/blob/master/optimization_notes_apple_m1.md) | Apple M1/M2 V0-V3 pipe (SDOT/SMMLA) analysis |
| [uops.info](https://uops.info/) · Intel optimization manual | Alder Lake / Raptor Cove port/lat/tp, basis for the memory-source broadcast |

---

## Code map

| area | file |
|---|---|
| public ABI | [`src/fe_pipeline.c`](../src/fe_pipeline.c) |
| frame pipeline | [`src/fe_engine.c`](../src/fe_engine.c) |
| STFT / iSTFT · FFT dispatch | [`src/fe_stft.c`](../src/fe_stft.c) · [`src/fe_fft.c`](../src/fe_fft.c) · [`src/fft/`](../src/fft/) |
| Conv1d wrappers · Winograd F(2,3) | [`src/nn/fe_conv.c`](../src/nn/fe_conv.c) · [`src/winograd/fe_winograd.c`](../src/winograd/fe_winograd.c) |
| GRU · MHSA · activations | [`src/nn/fe_gru.c`](../src/nn/fe_gru.c) · [`src/nn/fe_attention.c`](../src/nn/fe_attention.c) · [`src/nn/fe_activations.c`](../src/nn/fe_activations.c) |
| int8 GEMM (ARM) | [`src/qgemm/arm/`](../src/qgemm/arm/) — `qgemm_neon.c` · `qgemm_dotprod.c` · `qgemm_i8mm.c` |
| int8 GEMM (x86) | [`src/qgemm/x86/`](../src/qgemm/x86/) — `qgemm_avx2.c` · `qgemm_avxvnni.c` · `qgemm_avx512vnni.c` |
| SIMD post-epilogue · activation quantize · dispatch | [`src/qgemm/qgemm_simd_post.inl`](../src/qgemm/qgemm_simd_post.inl) · [`qgemm_quant.c`](../src/qgemm/qgemm_quant.c) · [`qgemm_dispatch.c`](../src/qgemm/qgemm_dispatch.c) |
| weight loader / blob format | [`src/fe_weights.c`](../src/fe_weights.c) |
