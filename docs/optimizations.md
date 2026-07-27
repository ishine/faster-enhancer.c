# 최적화 설계 노트

<sub>W8A8 양자화 · 6-tier int8 GEMM 마이크로커널 · fp16 cross-stage storage · op fusion · 측정 방법론.<br>
모델 자체의 spec(shape · 수식 · 레이어 인벤토리)은 <a href="architecture.md">architecture.md</a>, canonical RTF/품질 표는 루트 <a href="../README.md">README</a>.</sub>

> **TL;DR** — 지배적 설계 제약은 성능이 아니라 **cross-tier byte-identity** 다. universal `[−127,127]` clamp, FMA3 강제(SSE4.1 drop), AVX-512 에서 12-bit `VRCPPS` 재현 같은 결정이 전부 여기서 파생됐다. 그 제약 위에서 각 tier 는 자기 ISA 의 최강 int8 dot 명령을 골라 **ISA peak 의 85–93 %** 에 도달한다.

---

## 1. W8A8 양자화

| 항목 | 디자인 |
|---|---|
| **Weight** | per-output-row **symmetric int8**, `scale = max\|w\|/127`, zero-point 0 |
| **Activation** | per-tensor **asymmetric u8**, **매 프레임 재계산** (ORT `DynamicQuantizeLinear` 의미) |
| **Storage** | activation 을 `int8 = u8 − 128` 로 저장 → XOR/shift 없이 signed×signed 커널 재사용 |
| **Clamp** | weight·activation 양쪽에 universal `[−127, 127]` |
| **Bias correction** | zp offset 을 dequant epilogue 의 fp32 항에 흡수:<br>`bias_eff = bias + scale_x · scale_w[n] · (128 − zp_x) · row_sums[n]` |
| **Calibration freeze** | 없음. `FeActScale` 은 진단용 running min/max 만 기록 |

dequant 는 `y[m,n] = scale_x · scale_w[n] · (C32[m,n] + (128 − zp_x)·row_sums[n]) + bias[n]` 이고, 실제로는 `combined_scale[n]` / `bias_eff[n]` 로 미리 접어 커널 epilogue 의 단일 FMA 로 처리한다.

### 설계 선택 근거

| 결정 | 근거 |
|---|---|
| **dynamic** activation (static 아님) | static calibration freeze 시 phoneme blur 회귀. per-frame min/max scan 은 작은 고정비 |
| **asymmetric u8** activation | symmetric 대비 2–16 kHz consonant 정보 회복. zp offset 은 bias 로 흡수되어 dot 비용 0 |
| **per-row symmetric** weight | row 별 dynamic range 보존. zp=0 이라 weight 측 row_sums 보정 불필요 |
| **W8** (sub-byte 안 함) | W7 는 AVX2 `vpmaddubsw` polyfill 의 saturation 누적으로 PESQ −0.12 |

모델은 대부분 weight 가 load-bearing 이다 — 10 % 0-shot magnitude pruning 만 걸어도 PESQ 1.66 → 1.19 로 실패한다. W8A8 가 quality 를 보존하는 최저 비트폭에 가깝다.

### universal `[−127, 127]` clamp

int8 표현 범위는 `[−128, 127]` 이지만 `−128` 을 버린다. 이 한 줄이 세 가지를 동시에 산다.

**(a) saturation-free dot** — 가장 빡빡한 i16 누적 경로에서도 overflow 가 수학적으로 불가능하다.

| 경로 | worst-case | 한계 | margin |
|---|---|---|---|
| AVX2 `vpmaddubsw` pair-sum | 127×127 + 127×127 = 32,258 | INT16_MAX 32,767 | 509 |
| NEON 2-step `vmlal_s8` chain | 2 × 127×127 = 32,258 | INT16_MAX 32,767 | 509 |

NEON i16 chain 이 정확히 2 K-step 으로 제한되는 이유이기도 하다 — 세 번째 `vmlal_s8` 은 overflow.

**(b) cross-tier byte-identity** — AVX2 는 native int8 dot 이 없어 `_mm256_sign_epi8` 로 부호를 옮기는데, 이 명령이 `−128` 을 negate 하지 못한다(`−(−128) = +128` 은 int8 표현 불가). `a = −128` 일 때 native dot 을 쓰는 다른 tier 와 1-LSB 어긋나 byte-id 가 깨진다. clamp 로 `−128` 을 제거하면 edge case 자체가 사라진다.

**(c)** 대가는 최음수 bin 의 ~1 LSB 정밀도이고, **PESQ Δ = 0.000** (3 dp).

> clamp 는 단일 tier 변경이 아니라 **모든 후속 quant trick 의 budget basis** 다.

### 품질

3 클립(babble/car/street @5 dB) · 5569-frame window · Apple M2 / I8MM · 2026-07-25 재측정. ARM family 는 byte-id 이므로 세 tier 가 같은 출력 품질을 공유한다.

**clean target 대비 absolute quality**

| variant | PESQ-WB | STOI | eSTOI | SNR dB |
|---|---:|---:|---:|---:|
| `noisy_5dB` | 1.120 | 0.8576 | 0.6607 | 4.98 |
| `onnx_fp32` | 1.833 | 0.9440 | 0.8721 | 15.07 |
| **`fe_q8`** | **1.837** | 0.9442 | 0.8726 | 15.01 |

**faster-enhancer.c q8 vs fp32 ONNX fidelity**

| 클립 | PESQ-WB | STOI | eSTOI | SNR dB | max\|d\| |
|---|---:|---:|---:|---:|---:|
| babble_5dB | 4.569 | 0.9991 | 0.9981 | 32.47 | 0.0357 |
| car_5dB | 4.595 | 0.9998 | 0.9996 | 36.06 | 0.0117 |
| street_5dB | 4.573 | 0.9991 | 0.9984 | 33.60 | 0.0399 |
| **평균** | **4.579** | 0.9993 | 0.9987 | **34.04** | 0.0291 |

<details>
<summary><b>reference 는 엔진과 같은 causal STFT grid 여야 한다</b> — 측정 방법론 정정 (2026-07-25)</summary>

<br>

이전 reference(center=True, 현재는 삭제)는 `torch.stft` 를 `center=` 인자 없이 호출했다 → PyTorch 기본값 `center=True`, 즉 `n_fft/2 = 512` reflect-pad 후 프레임이 `t·hop` 에 **중심**을 둔다. 반면 C 엔진은 인과 streaming STFT 로 프레임 t 가 `[t·hop − 704, t·hop + 320)` 를 덮는다(`center=False` + 704-sample zero left-history).

두 analysis grid 가 `704 − 512 = 192` 샘플 어긋나고, 192 는 hop 320 의 배수가 아니라 상쇄되지 않는다. **같은 fp32 모델**을 192 샘플 어긋난 두 grid 로 돌려 비교하면 그 자체로 SNR ≈ 22.8 dB · PESQ-WB ≈ 4.30 이 나온다 — 과거에 "q8 vs fp32 fidelity" 로 보고하던 값(4.253 / 24.73 dB)과 구분되지 않는 수준이다. 즉 그 표는 int8 양자화 오차가 아니라 **프레이밍 위상차**를 재고 있었다.

`tools/onnx_reference_causal.py` 는 `src/fe_stft.c` 의 프레이밍·합성 window 정규화를 그대로 복제한다. 정합 후 q8 출력은 reference 와 **lag 0** 으로 정렬되며(이전에는 704 필요), fidelity 는 PESQ 4.253 → **4.579**, SNR 24.73 → **34.04 dB** 로 올라간다. 정정 방향이 프로젝트에 유리하다 — 이전 값은 실제 양자화 충실도를 과소평가하고 있었다.

절대 품질 표는 이 정정의 영향을 거의 받지 않는다: causal grid fp32 ONNX 는 PESQ 1.833(이전 non-causal 1.835), 즉 `center=True` 가 갖는 512-sample look-ahead 이점이 이 태스크에서는 무시할 수준이다.

</details>

---

## 2. GEMM 마이크로커널 (6 tier)

각 tier 는 같은 fused-kernel family 를 구현한다. scalar fallback 은 없다 — `fe_qgemm_init` 이 baseline ISA 를 검사하고 미달이면 `fe_init` 이 non-zero 를 반환한다(silent slow-path 제거).

| tier | 최강 int8 dot | MAC/inst | tile | ISA-peak | 대상 |
|---|---|---:|---|---:|---|
| **NEON** (floor) | `vmull_s8`→`vmlal_s8`→`vpadalq_s16` | 8 | 4×8 | ~88 % | arm64 baseline |
| **DOTPROD** | `vdotq_laneq_s32` | 16 | 8×8 | ~85–87 % | arm64 + FEAT_DotProd |
| **I8MM** | `vmmlaq_s32` (2×8·8×2→2×2) | 32 | 8×8 | ~87 % | arm64 + FEAT_I8MM |
| **AVX2** (x86 floor) | `vpmovsxbw` + `vpmaddwd` | ~24 | 12×8 | ~90 % | Haswell+ (AVX2+FMA3+F16C) |
| **AVX-VNNI** | `vpdpbusd ymm` | 32 | 12×8 | ~93 % | Alder Lake+, Zen 4 |
| **AVX-512-VNNI** | `vpdpbusd zmm` | 64 | 12×16 | zmm-full | Ice Lake / SPR, Zen 4 |

모든 tier 가 구현하는 커널 집합: `int32` / `int32_k20`† / `fp32_fused` / `fp32_fused_acc` / `fp32_fused_track_maxabs` / `gru_full_fused_fp16inout`.

† `_int32_k20`(MHSA Q@Kᵀ 전용 K=20 완전 unroll)은 **I8MM 만 미구현**이다. K=20 이 8 의 배수가 아니라 SMMLA k8-group 에 안 떨어져, I8MM 하드웨어에서도 attention score GEMM 은 매 프레임 `qgemm_dotprod_int32_k20` 으로 routing 된다 — remainder fallback 이 아니라 **steady-state 경로**다.

### tier 별 핵심 기법

**NEON** — ARMv8.0 엔 int8 dot 이 없다. `vmull_s8` 로 시작해 두 K-step 을 i16 누산기에 `vmlal_s8` 로 chain 한 뒤 `vpadalq_s16` 한 번으로 i32 promote(ggml q3_K 패턴). 4×8 microkernel 의 핵심은 A 의 4-byte K-quad 를 **행당 한 번만** `vld1_dup_s32` 로 load 해 cols 0-7 전부에 재사용하는 것 — 이전 8×4-twice 경로 대비 A-side LD1R traffic 절반. register-resident peak 은 `vmull+vpadalq ~55 GMAC/s` 로 구조적으로 낮다.

**DOTPROD** — 병목은 dot 명령이 아니라 **A-operand load** 였다. legacy 경로는 SDOT 의 `a` 를 매 행·매 k4-group 마다 `vld1q_dup_s32`(LD1R)로 재구성해, M2 에서 SDOT peak 의 ~62 %(136 GMAC/s)로 throttle 됐다. **A row-quad pre-pack + lane-indexed dot** 으로 해소: A 를 한 번 repack 해 각 k4-group 이 8 행의 4-byte quad 를 연속으로 담게 하고, `vld1q_s8` 두 개만 load 해 `vdotq_laneq_s32` 의 lane 0..3 으로 먹인다 — LD1R 전부 제거, ~186 GMAC/s(peak 의 85–87 %).

**I8MM** — SMMLA 의 `a` 피연산자는 16 B = `[r0_K0..7 | r1_K0..7]`. legacy 는 매 K-block 마다 8 `vld1_s8` + 4 `vcombine` = 12 µop 으로 조립했다. **A row-pair pre-pack**(KleidiAI-style)으로 동일 layout 을 미리 만들면 `vld1q_s8_x4` 한 µop 이 4 row-pair 전부를 커버 — K-block 당 8 µop 절감. epilogue 는 2×2 SMMLA 출력을 `vtrn1q/vtrn2q_s64` 로 행우선 gather. ~183 GMAC/s.

**AVX2** — 고전적 `vpmaddubsw` 는 saturate 가능해 못 쓴다(u8×i8 pair-sum max `2×255×127 = 64,770 > 32,767`). 현재 경로는 `vpmovsxbw` + `vpmaddwd`(XNNPACK-style): i16×i16 → i32 pair-sum 이라 saturate 가 원천적으로 불가능한 유일한 bit-id-safe path. B 는 N-block 당 한 번 K-pair-interleaved 로 re-pack, A 는 호출당 한 번 i16 으로 pre-expand. **ISA floor**: per-K-quartet 8 `vpmaddwd` on p01 = 8 cyc, AVX-VNNI(4 cyc)의 정확히 2배 — 이게 bit-id-safe AVX2 의 본질적 한계다.

**AVX-VNNI** — `vpdpbusd` 가 (u8, i8) 을 요구하므로 A 전체를 호출당 1회 `XOR 0x80` 한 뒤(`prexor128_a`) inner loop 은 broadcast + vpdpbusd 만 수행하고, 보정항 `wsum[n] = 128·Σ_K b[n,k]` 를 마지막에 뺀다. 세 기법:

- **memory-source broadcast** — `_mm256_set1_epi32(*(int32_t*)p)` 는 `vmovd; vpbroadcastd ymm,xmm` 로 lowering 되어 **p5 만** 점유하는데, vpdpbusd 도 p0+p5 라 dot 슬롯 절반을 뺏는다. `_mm256_broadcastd_epi32(_mm_loadu_si32(p))` 로 강제하면 `vpbroadcastd ymm,[mem]` 한 µop(load port)으로 emit.
- **12-row tile** — vpdpbusd 는 lat 5 / tput 2. 8 chain 은 8/5 = 1.6/cyc 로 latency-bound(peak 의 80 %), 12 chain 이면 dispatch 포화(~93 %). reg budget 14 ymm, spill 없음.
- **wsum 4-way chain-split** — 단일 누산기면 chain depth = k4_groups·5 (K=72 → 90 cyc vs dispatch 18 cyc). 4분할로 depth ~25 cyc. 정수 덧셈 결합법칙 + saturation-free 라 bit-id 유지.

M-tail 도 SIMD 를 유지해야 한다 — 공유 `qgemm_scalar_int32` 는 prexor 된 u8 A 를 signed 로 읽고 wsum 보정도 안 해 **틀린 값**을 낸다. Winograd M=64 는 `(M/24)*24` 로 12-loop 을 돌려 나머지를 8-align(`4×12 + 2×8`).

**AVX-512-VNNI** — tier 전체가 zmm 16-lane. 매-프레임 activation quantize 도 `target("avx512f,avx512bw,avx512vl")` clone 으로 zmm 화(`fe_qg_x86_avx512` flag 로 런타임 dispatch) — 16-wide GEMM 을 8-wide quantize 가 먹이던 잠복 perf 버그 해소. N%16==8 leftover 는 masked-zmm 이 아니라 ymm helper 로(lane 낭비 회피).

> **cross-tier bit-id 버그 (Intel SDE 로 발견)** — avx512vnni 출력이 avx2/avxvnni 와 SNR 44 dB 로 byte-identical 이 **아니었다**. 원인은 sigmoid/tanh 의 reciprocal 정밀도: ymm 은 `_mm256_rcp_ps`(~12-bit), zmm 은 `_mm512_rcp14_ps`(14-bit) → GRU gate / SiLU 누적 divergence. `fe_avx512_rcp_match256` 이 zmm den 을 두 ymm 반쪽으로 쪼개 `_mm256_rcp_ps` 로 12-bit reciprocal 을 동일 bit 로 재현한다(AVX-512F only).

### ISA floor / ceiling

Winograd k=3 path 의 dense-equivalent runtime multiply footprint 는 **46.3 MMAC/frame**(direct-conv equivalent 53.4). 6.667 ms budget, ~3.8 GHz 기준 x86:

| tier | peak MAC/cyc | GEMM-only floor | +overhead 현실 floor |
|---|---:|---:|---:|
| AVX-VNNI | 64 | ~0.029 | **~0.04** |
| AVX2 | ~24 | ~0.076 | **~0.09** |
| AVX-512-VNNI | 128 | ~0.014 | **~0.03** |

AVX2 0.07 은 floor 아래고, 0.03 의 경로는 AVX-VNNI(ymm)가 아니라 AVX-512 zmm tier 다.

**Apple M2 ISA ceiling** — register-resident 실측 peak SMMLA **211 GMAC/s**, SDOT **220**. `vdotq_s32` 는 4 pipe × 16 = 64 MAC/cyc, `vmmlaq_s32` 는 2 pipe × 32 = 64 — I8MM 의 2× MAC density 가 dispatch port 절반으로 **정확히 상쇄**된다. 그래서 M2 에서 DOTPROD GEMM(186) ≈ I8MM(183). M2 에선 SIMD 영역 추가 win 이 없고, 남은 영역은 다른 ARM 코어(vmmlaq 4-pipe)나 algorithm-level 이다.

전 tier hot tile 은 register spill 이 0 이다 — intrinsics + target-clone 이 컴파일러에 레지스터 할당·스케줄 자유를 주므로 hand-asm 은 이득 없이 per-uarch 유지보수만 늘린다.

**BSS pre-fault** — AVX2 i16-expand 경로의 363,520-byte scratch 가 first-touch page fault 로 peak/p50 ratio 를 5.78× 로 만들었다. `fe_qgemm_init` 에서 모든 page 를 한 번 touch(`qgemm_avx2_prefault_buffers`)해 정상 경로 밖으로 뺐다 — peak −66 %, ratio 2.1×. ARM 두 tier 도 동일(`a_rp` + GRU 전용 `gru_x_rp`·`gru_h_rp`).

---

## 3. fp16 cross-stage storage

레이어 간 long-lived state 를 IEEE binary16 으로 저장한다. 대상은 정확히 두 버퍼 — `enc_skip`(encoder→decoder skip)과 `gru_h`(recurrent state). `attn_*` 같은 intra-frame scratch 는 매 호출 zero-re-init 이라 제외.

| 버퍼 | fp32 | fp16 | 절감 |
|---|---:|---:|---:|
| `enc_skip[4]` | 196,608 B | 98,304 B | −98,304 B |
| `gru_h[4]` | 82,944 B | 41,472 B | −41,472 B |
| `enc_skip_scratch` (staging, 삭제됨) | 49,152 B | 0 B | −49,152 B |
| **Named state/staging delta** | | | **−188,928 B (184.5 KiB)** |

현재 `sizeof(FeState) = 432,384` bytes.

int8 storage 는 기각됐다 — `dec_1x1` 의 joint min/max round-trip 에서 SNR 을 ~40 dB 잃었다. fp16 은 mantissa 10 bit 를 유지한다.

**GRU dual-store ngate epilogue** — 커널이 직전 hidden state 를 fp16 slot 에서 직접 읽고(`fe_fp16_load4`), 업데이트된 state 를 fp32(즉시 소비자 `rnn_fc`)와 fp16(다음 프레임 `gru_h`)으로 **동시에** 쓴다. 이전 fp32-inout path 가 필요로 하던 GRU 앞 unpack + 뒤 pack 패스를 둘 다 제거한다.

**fp16 변환은 mandatory** — x86 은 F16C(`VCVTPS2PH`/`VCVTPH2PS`), aarch64 는 `FCVTN`/`FCVTL`. scalar fallback 은 빌드에 없고, 미지원 host 는 `fe_init` 실패로 처리한다(조용히 안 도는 위험 회피).

**Page hygiene** — `fe_init` 은 warmup **직전** 에 blob + weight 버퍼 + `FeState` 를 best-effort page-lock(`mlock`/`VirtualLock`)한다. lock 을 warmup 보다 먼저 거는 이유는 warmup 자체가 회피하려던 soft fault 를 만들지 않게 하기 위함이다. 이어 128-frame 무음 warmup 이 weight page·BSS scratch·branch predictor·L1/L2 를 데운다. activation scale 은 warmup 에서 freeze 되지 않는다.

**Working-set** — Winograd runtime scratch 는 `wino_U`(98,304 B) + `wino_dq`(24,576 B) = 122,880 B 로 L2-resident 로 본다. weight-derivation scratch 258,048 B 는 init-time 에 `malloc`/`free` 되어 resident 하지 않는다.

**Build-flag 위생** — x86 은 tier 별 per-file `-march` 외에, 10 개 TU 목록(`FE_X86_AVX2_TUS`)에 `-mavx2 -mfma -mf16c` 를 따로 붙인다. 이 목록에서 빠지면 해당 TU 의 SIMD 블록이 전처리 단계에서 **조용히 사라진다** — `fe_winograd.c` 에서 실제로 발생했던 잠복 버그다.

---

## 4. Op fusion

인접 op 를 합쳐 fp32 중간 버퍼 round-trip / 별도 transpose / 별도 pack 을 제거한다. *dequant epilogue → 다음 op 의 input quantize* 같은 stage boundary 가 prime target.

| fusion | 내용 |
|---|---|
| **transpose + quantize** | freq 축 Linear 의 `[F,C] → [C,F]` transpose 를 int8 quantize pass 안에 fold (4×4 fp32 tile transpose + saturating narrow) |
| **residual in epilogue** | `resA`/`resB` 를 별도 add pass 대신 `rnn_fc`/`attn_fc` dequant epilogue 의 accumulating store(`C += …`)로 흡수 |
| **bias_eff** | zp offset 보정을 bias 에 미리 접어 dot 비용 0 |
| **GRU full-fusion** | `W_ih·x` + `W_hh·h` + r/z/n gate + h-update 를 per-tier 단일 커널로. int32 누산기가 메모리로 안 떨어지고 L1d-resident `ih_tile` 에 안착 |
| **Winograd batched epilogue** | 4× int32 U → dequant + Aᵀ + bias + optional SiLU + fp16 skip write 를 한 pass 로 |
| **softmax-from-int32** | int32 score 를 직접 받아 dequant→softmax→int8 을 2-pass 로. 전체 fp32 score matrix 를 materialize 하지 않고 row scratch 만 사용 |
| **max-abs tracking** | 다음 layer 의 재양자화 scale 용 max\|y\| 를 별도 스캔 대신 dequant-store 중 레지스터로 추적 |
| **concat + quantize** | decoder 1×1 이 fp32 A 와 fp16 B 의 min/max 를 합쳐 하나의 (scale, zp) 로 같은 aq 행에 직접 emit |
| **STFT window pre-fold** | iFFT 의 1/N 을 합성 window 에 미리 접어 커널에서 제거 |

**Winograd F(2,3)** — encoder/decoder 의 k=3 conv(Cin=Cout=96)를 Winograd 로 실행해 곱셈을 **2/3 로** 줄인다(direct `3·Ci·Co·F` → `4·(F/2)·Co·Ci`). 입력 변환은 **two-pass** 다: pass A 가 Bᵀ 를 계산하며 4 component 별 max-abs 만 뽑고 결과를 버리고, pass B 가 Bᵀ 를 다시 계산하면서 확정된 scale 로 바로 int8 양자화한다 — 4개 fp32 변환 결과(98 KiB)를 L1d 에 올리지 않으려 재계산이 메모리 트래픽보다 싸다고 판단한 것. 변환 행렬이 ±1·±0.5 뿐이라 런타임 변환은 add/sub 만으로 끝난다(0.5 는 오프라인 weight 변환에만 등장). per-tile fused microkernel 은 프로토타입 후 폐기됐다 — MR=2 에서 weight reuse 를 잃어 측정한 모든 tier 에서 batched 경로보다 느렸다.

**fusion 의 한계** — production k=3 경로는 Winograd 전용이고 direct/im2col k=3 경로가 없다. 진짜 im2col 은 `enc_pre` strided conv 하나뿐인데, 이건 `K = 2·Ci = 16` 이 너무 작아 int8 이 amortize 안 되어 fp32 sgemm 으로 남겼다.

---

## 5. 측정 방법론

### byte-identity 회귀 게이트

같은 입력에 대해 **architecture family 내** 모든 SIMD tier 가 byte-identical(SHA256 동일) 출력을 낸다: x86 `AVX2 ≡ AVX-VNNI ≡ AVX-512 VNNI`, ARM `NEON ≡ DOTPROD ≡ I8MM`.

성립 조건은 `-ffast-math` 없음, family 내 동일 계수·동일 reduction order, universal clamp. **Cross-architecture**(x86 ↔ ARM)는 IEEE round-tie 거동 차이로 byte-id 가 아니며 STOI/SNR 동등성으로 assert 한다.

| 방법 | 대상 | 내용 |
|---|---|---|
| `tools/sha256_matrix.sh` | 실행 가능한 모든 tier | 3 클립 × n tier WAV 해시. family 내 행끼리 동일 = PASS. pre/post diff 가 비면 회귀 없음 |
| `tests/fe_test_cross_tier.c` | 빌드된 모든 tier | cross-tier SNR 진단 + final float bit-id memcmp |
| Intel SDE (`-spr`) | AVX-512 VNNI | dev box 가 AVX-512 fuse-off 이라 기능 검증만. timing 은 불가 |

> **회귀 워크플로** — 자명하지 않은 코드 변경 전후로 `sha256_matrix.sh` 를 돌려 `diff`. empty diff = byte-level 회귀 없음.

SSE4.1 은 FMA3 부재로 dequant epilogue 가 `mul → add` 2단계 rounding → ~1 ULP/op drift → 누적 시 byte-id 붕괴로 영구 drop. x86 floor tier 이름은 AVX2 이되 런타임 요구는 **AVX2+FMA3+F16C** 다.

### 품질 측정

fp32 ONNX reference 는 **엔진과 동일한 causal STFT grid** 에서 생성한다(`tools/onnx_reference_causal.py`) — 근거는 §1 품질의 접힌 절 참고. `tools/quality_compare.py` 가 cross-correlation 으로 integer lag 를 추정해 정렬한 뒤 PESQ-WB / STOI / eSTOI / SNR 을 계산한다. grid 가 맞으면 lag 는 0 이다.

### hybrid-CPU latency tail

i5-13400F(6 P + 4 E)에서 default priority 로 재면 p50→p95 는 tight(~1.2×)한데 p99→peak 가 터지고, 단일 peak frame 이 **run-to-run 으로 크게 흔들린다**(VNNI max 0.89 ms → 다음 run 3.94 ms). 이 불안정성이 tell 이다 — deterministic 효과라면 안정적일 것이다.

pinning 으로 원인이 분리된다 (babble 5569 frames, VNNI, **ms/frame**):

| pin | p50 | p95 | p99 | peak | note |
|---|---:|---:|---:|---:|---|
| **single P-core** (LP4) | 0.403 | 0.416 | 0.437 | 0.95 | **p99 = 1.08× p50** — flat |
| **single E-core** (LP12) | **1.435** | 1.495 | 1.538 | 2.96 | **3.6× 느린 p50** |
| all P-core, normal prio | 0.382 | 0.496 | 0.642 | 2.17 | E 회피, 그래도 preempt |
| unpinned, normal prio | 0.385 | 0.506 | 0.697 | 3.96 | baseline tail |

두 사실이 떨어진다. (1) P-core 위에서 compute 는 본질적으로 jitter-free — 분리된 kernel-side tail 이 없다. (2) Gracemont E-core 는 256-bit VNNI 포트가 1개(Raptor Cove 는 2개)라 같은 frame 을 3.6× 느리게 돈다. 스케줄러가 E-core 에 park 한 *모든* frame 이 p95–p99 tail 샘플이 된다.

**레버는 affinity 가 아니라 thread priority 다:**

| config | p50 | p99 | peak |
|---|---:|---:|---:|
| normal priority | 0.378 | 0.633 | 1.4–5.2 |
| **`THREAD_PRIORITY_TIME_CRITICAL`** | 0.40 | **0.50** | **0.6–1.0** |
| single-P-core pin, *normal* prio | 0.40 | 0.918 | 6.45 ← 더 나쁨 |

Win11 Thread Director 는 high-priority 스레드를 스스로 P-core 로 steer 하므로 priority bump 하나로 E-core 회피가 공짜로 따라온다. 반대로 normal prio 의 single-core pinning 은 *더 나쁘다* — OS 가 그 코어에 뭘 스케줄하면 fall back 할 P-core 가 없다.

**왜 라이브러리 밖인가** — thread scheduling 은 integrator 의 몫이고, 호출자의 priority 를 몰래 올리는 라이브러리는 least-surprise 를 위반한다. 엔진은 priority 와 무관하게 byte-identical 이며 바뀌는 건 jitter envelope 뿐이다. 라이브러리가 하는 유일한 OS hygiene 은 page-lock 과 warmup 이고, 둘 다 guarded·실패 시 no-op 이다.

```c
// Windows (가장 강한 보장은 MMCSS AvSetMmThreadCharacteristics L"Pro Audio")
SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);
// macOS — QoS 가 thread 를 P-core 에 유지, 권한 불요
pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
// Linux — RT-capable 이면 SCHED_FIFO, 아니면 harmless no-op
struct sched_param sp = { .sched_priority = 50 };
pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp);
```

`fe_run_file` 은 platform-appropriate call 을 default 로 적용하고, `FE_RT=0` 으로 normal-priority envelope 을 비교 측정할 수 있다.

---

## 수치의 출처 구분

이 문서의 수치는 두 종류다. 공개 레포를 받아 확인하는 사람이 헷갈리지 않도록 구분해 둔다.

**`experiments/` 에 아티팩트가 있는 것** — 러너 로그·요약·raw CSV 가 커밋되어 있고 재현 스크립트가 함께 있다:
M2 tier/system/Winograd/build-option ablation, qgemm 프로파일, ONNX 대비 벤치,
VoiceBank-DEMAND 824 발화 품질, M2 에너지 매트릭스, M2 30분 paced sustained,
S23+ tier·placement·sustained.

**개발 중 측정했지만 아티팩트를 남기지 않은 것** — 아래 수치는 이 문서의 서술로만 남아 있다.
재측정 전까지는 논문에 인용하지 않는다:

| 수치 | 내용 |
|---|---|
| PESQ 1.66 → 1.19 | 10 % 0-shot magnitude pruning. 실험 스크립트·로그 없음 |
| W7 −0.12 PESQ | AVX2 `vpmaddubsw` polyfill saturation. 측정 스크립트도 함께 삭제됨 |
| hybrid-CPU latency tail 표 | i5-13400F LP4/LP12 pinning. `experiments/` 에 x86 디렉터리 자체가 없음 |
| BSS pre-fault peak/p50 5.78× → 2.1× | page-fault jitter 측정. 로그 없음 |

Table 1 의 x86 행(AVX-VNNI 0.057 / AVX2 0.110)도 같은 상태다 — 현재 repeat 프로토콜
이전의 측정이고 아티팩트가 없다. 논문 limitations 에 그렇게 적혀 있다.

## Sources

| 출처 | 기법 |
|---|---|
| [ggml q3_K / tq1_0](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cpu/arch/arm/quants.c) | NEON `vmull→vmlal_s8 chain→vpadalq_s16` i16-level fusion |
| ggml / llama.cpp Q8_0 | AVX2 vpsignb saturation-free dot (현재는 vpmaddwd i16-expand 로 대체) |
| [XNNPACK](https://github.com/google/XNNPACK) `qd8-f32-qc8w-gemm-4x8c8-avx2` | AVX2 `vpmovsxbw + vpmaddwd` 3-op dot |
| ARM KleidiAI | I8MM A row-pair pre-pack layout |
| [insn_bench_aarch64](https://github.com/ocxtal/insn_bench_aarch64/blob/master/optimization_notes_apple_m1.md) | Apple M1/M2 V0-V3 pipe (SDOT/SMMLA) 분석 |
| [uops.info](https://uops.info/) · Intel optimization manual | Alder Lake / Raptor Cove port·lat·tp, memory-source broadcast 근거 |

---

## 코드 위치

| 영역 | 파일 |
|---|---|
| 공개 ABI | [`src/fe_pipeline.c`](../src/fe_pipeline.c) |
| 프레임 파이프라인 | [`src/fe_engine.c`](../src/fe_engine.c) |
| STFT / iSTFT · FFT dispatch | [`src/fe_stft.c`](../src/fe_stft.c) · [`src/fe_fft.c`](../src/fe_fft.c) · [`src/fft/`](../src/fft/) |
| Conv1d wrappers · Winograd F(2,3) | [`src/nn/fe_conv.c`](../src/nn/fe_conv.c) · [`src/winograd/fe_winograd.c`](../src/winograd/fe_winograd.c) |
| GRU · MHSA · activations | [`src/nn/fe_gru.c`](../src/nn/fe_gru.c) · [`src/nn/fe_attention.c`](../src/nn/fe_attention.c) · [`src/nn/fe_activations.c`](../src/nn/fe_activations.c) |
| int8 GEMM (ARM) | [`src/qgemm/arm/`](../src/qgemm/arm/) — `qgemm_neon.c` · `qgemm_dotprod.c` · `qgemm_i8mm.c` |
| int8 GEMM (x86) | [`src/qgemm/x86/`](../src/qgemm/x86/) — `qgemm_avx2.c` · `qgemm_avxvnni.c` · `qgemm_avx512vnni.c` |
| SIMD post-epilogue · activation quantize · dispatch | [`src/qgemm/qgemm_simd_post.inl`](../src/qgemm/qgemm_simd_post.inl) · [`qgemm_quant.c`](../src/qgemm/qgemm_quant.c) · [`qgemm_dispatch.c`](../src/qgemm/qgemm_dispatch.c) |
| weight 로더 / blob 포맷 | [`src/fe_weights.c`](../src/fe_weights.c) |
