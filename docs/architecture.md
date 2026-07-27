# FastEnhancer‑Medium 48 kHz — 레이어 reference

<b>엔진에 배포된 Conv / GRU / Attention 레이어의 단일 self‑contained 설명.</b><br>
<sub>"모델이 무엇인가"의 권위 있는 출처 — "어떻게 최적화했나"는 <a href="../README.md">README</a> + <a href="optimizations.md">optimizations.md</a>.</sub>

> <b>TL;DR</b> — 48 kHz monaural denoiser. STFT → encoder(F 축 ×4 downsample) → RNNFormer × 4 → decoder(×4 upsample) → iSTFT. RNNFormer block = <code>GRU → rnn_fc(+res) → MHSA(+res)</code>, 전부 <b>frequency 축</b>에서 동작. W8A8 int8, 511,754 params, 320-sample processing quantum(6.67 ms), zero look‑ahead, fixed STFT alignment delay 704 samples(14.67 ms).

---

## 모델 한눈에

<table>
<thead><tr><th align="left">항목</th><th align="left">값</th></tr></thead>
<tbody>
<tr><td>Task</td><td>실시간 monaural 노이즈 서프레션</td></tr>
<tr><td>Sample rate</td><td>48 kHz</td></tr>
<tr><td>Frame / hop</td><td>320 sample = 6.67 ms</td></tr>
<tr><td>STFT</td><td>n_fft = 1024, <code>fe_run()</code> 호출 간 overlap‑add 유지</td></tr>
<tr><td>Latency / alignment</td><td>zero look‑ahead; fixed STFT alignment delay 704 samples (14.67 ms), one call emits 320 samples</td></tr>
<tr><td>Params</td><td>511,754 (565,108-byte / 551.9 KiB q8 blob; SHA-256 <code>d6f98e186bee6ed655b52a2edf5f707ccf761c755bb69de8de7f68cad42d7837</code>; 2,047,016-byte / 1.95 MiB fp32-equivalent)</td></tr>
<tr><td>Working tensors</td><td><code>[freq, channels]</code> row‑major 전체</td></tr>
<tr><td>Channels</td><td>encoder / decoder C1 = 96 · RNNFormer C2 = 72</td></tr>
<tr><td>Freq dims</td><td>enc_pre 후 F1 = 128 · RNNFormer 내부 F2 = 72</td></tr>
<tr><td>Heads</td><td>NH = 4, HD = C2 / NH = 18</td></tr>
<tr><td>Nonlinearity</td><td>SiLU 만 (attention softmax · GRU sigmoid/tanh 는 별도)</td></tr>
</tbody>
</table>

<div align="center">

<code>audio → STFT → encoder (F 축 stride 4 downsample) → RNNFormer × 4 → decoder (F 축 stride 4 upsample) → iSTFT → audio</code>

</div>

<b>입력 압축:</b> magnitude 를 <code>^0.3</code> 으로 압축해 네트워크 입력, 출력 시 <code>^(1/0.3)</code> 역변환. 위상은 그대로 통과 — 네트워크는 magnitude mask 만 출력해 입력 위상에 곱한다.

<b>양자화:</b> W8A8 — per‑row symmetric int8 weights, per‑tensor asymmetric uint8 activations 를 <b>매 프레임 재계산</b>(dynamic quantize). 요약은 <a href="#양자화-요약">아래</a>, 전체 디자인은 <a href="optimizations.md">optimizations.md</a>.

<b>Cross‑stage state:</b> recurrent / skip 버퍼 모두 fp16 패킹 저장 (<a href="#cross-stage-storage-fp16">아래</a>).

---

## Convolutions

전체 convolution 은 모두 <b>frequency 축</b>으로 적용되는 1‑D Conv 다 (시간 축 아님). 입력 <code>[freq=F, Cin]</code> → 출력 <code>[freq=F, Cout]</code>.

### 레이어 인벤토리

<table>
<thead><tr>
<th align="left">Layer</th><th>Kernel</th><th>In ch</th><th>Out ch</th>
<th align="left">Weight (Cout × Cin × k)</th><th align="left">Path</th><th>양자화</th>
</tr></thead>
<tbody>
<tr><td>enc_pre (strided)</td><td>8, s4</td><td>2 (re+im)</td><td>C1=96</td><td>96 × 8 × 2 <sub>(reshape trick)</sub></td><td>reshape → Conv1d(8→96, k=2)</td><td>fp32 only</td></tr>
<tr><td>enc[0..2]</td><td>3, s1, p1</td><td>96</td><td>96</td><td>96 × 96 × 3</td><td><b>Winograd F(2,3)</b></td><td><b>int8</b></td></tr>
<tr><td>rf_pre_conv</td><td>1</td><td>96</td><td>C2=72</td><td>72 × 96 × 1</td><td>GEMM</td><td><b>int8</b></td></tr>
<tr><td>rf_post_conv</td><td>1</td><td>72</td><td>96</td><td>96 × 72 × 1</td><td>GEMM</td><td><b>int8</b></td></tr>
<tr><td>dec[i].1x1 (×3)</td><td>1</td><td>2·C1=192</td><td>96</td><td>96 × 192 × 1</td><td>GEMM</td><td><b>int8</b></td></tr>
<tr><td>dec[i].3x3 (×3)</td><td>3, s1, p1</td><td>96</td><td>96</td><td>96 × 96 × 3</td><td><b>Winograd F(2,3)</b></td><td><b>int8</b></td></tr>
<tr><td>dec_post_1x1</td><td>1</td><td>2·C1=192</td><td>96</td><td>96 × 192 × 1</td><td>GEMM</td><td><b>int8</b></td></tr>
<tr><td>dec_post_up (transpose)</td><td>8, s4</td><td>96</td><td>2 (re+im)</td><td>96 × 2 × 8</td><td>hand‑tuned</td><td>fp32 only</td></tr>
</tbody>
</table>

3 개 encoder block 뒤에 대응하는 3 개 decoder block 이 concat‑style skip connection 으로 연결된다 — encoder 출력이 upsample 된 decoder feature 와 채널 축으로 concat (채널 수 2·C1) → <code>dec[i].1x1</code> 이 다시 C1 으로 mix.

### k=3 conv — Winograd F(2,3)

encoder / decoder 의 k=3 conv (Cin=Cout=96) 는 <b>Winograd F(2,3)</b> 로 실행해 direct im2col 대비 MAC 을 <b>−33 %</b> 줄인다.

<table>
<thead><tr><th align="left">항목</th><th align="left">내용</th></tr></thead>
<tbody>
<tr><td>Pass 구조</td><td><b>Pass A</b> (vmax‑only, no storage) + <b>Pass B</b> (re‑read + per‑component scale + int8 quantize)</td></tr>
<tr><td>Inner GEMM</td><td>4× <code>gemm_int32</code> (dequant epilogue 생략)</td></tr>
<tr><td>Epilogue</td><td>batched fused — 4× int32 U → dequant + A<sup>T</sup>(even sum, odd signed sum) + bias + opt SiLU 를 한 pass 로</td></tr>
</tbody>
</table>

Output 은 fp32, 다음 layer 의 quantize 패스가 dynamic min/max 로 다시 양자화. 가중치는 논리적으로 <code>[Cout, Cin, k]</code> 이지만 packed int8 + output row 당 fp32 scale 1개로 저장. production q8 loader 는 모든 k=3 layer 의 Winograd weight 를 로드 직후 파생하며, 이 단계가 실패하면 로드 실패로 처리한다.

### k=1 conv

k=1 conv = freq 별 linear layer. 입력 <code>[F, Cin]</code> → 출력 <code>[F, Cout]</code>, weight <code>[Cout, Cin]</code> — 단일 W8A8 GEMM + bias + 옵션 SiLU + 옵션 int8 출력. decoder concat 1×1 은 <code>[F, 2·C1]</code> 타일(encoder skip + 현재 decoder feature 채널 축 concat)을 <code>[F, C1]</code> 으로 project.

### fp32‑only 레이어

<table>
<thead><tr><th align="left">Layer</th><th align="left">이유</th></tr></thead>
<tbody>
<tr><td><b>enc_pre</b></td><td>StridedConv reshape trick 이 2채널(re, im) 입력을 freq 해상도 1/4 의 Ci'=8 채널로 재배치 → k=2 conv. 입력 분포 특이성 때문에 fp32 유지.</td></tr>
<tr><td><b>dec_post_up</b></td><td>마지막 ConvTranspose1d(kernel 8, stride 4)가 <code>[2, F=512]</code> 로 upsample. fp32 유지.</td></tr>
</tbody>
</table>

---

## GRU

### 위치

4개 RNNFormer block 각각이 GRU step 1개씩 포함. <b>freq 축</b>으로 적용 — 각 freq bin 이 scan 의 "timestep" 역할을 하고, hidden state 는 freq bin 별로 보유하며 <b>오디오 프레임 간 유지</b>된다.

### Shape & 수식

- Hidden dim <b>D = 72</b> (= C2).
- Gates: 표준 PyTorch <code>[r, z, n]</code>, linear‑before‑reset (<code>r</code> 이 hidden‑side <code>n</code>‑gate 입력을 mask).

<div align="center">

<table>
<tbody>
<tr><td align="left"><code>r&nbsp;= sigmoid(W_ir·x + b_ir + W_hr·h + b_hr)</code></td></tr>
<tr><td align="left"><code>z&nbsp;= sigmoid(W_iz·x + b_iz + W_hz·h + b_hz)</code></td></tr>
<tr><td align="left"><code>n&nbsp;= tanh&nbsp;&nbsp;&nbsp;( (W_in·x + b_in) + r · (W_hn·h + b_hn) )</code></td></tr>
<tr><td align="left"><code>h' = n + z · (h − n)</code></td></tr>
</tbody>
</table>

</div>

가중치는 <b>fused 두 행렬</b>로 저장한다.

<table>
<thead><tr><th align="left">행렬</th><th align="left">Shape</th><th align="left">행 순서</th></tr></thead>
<tbody>
<tr><td><code>W_ih</code> (input‑side)</td><td>[3D, D] = [216, 72]</td><td><code>[W_ir; W_iz; W_in]</code></td></tr>
<tr><td><code>W_hh</code> (hidden‑side)</td><td>[3D, D] = [216, 72]</td><td>동일</td></tr>
</tbody>
</table>

bias <code>b_ih</code>, <code>b_hh</code> 각각 <code>[3D]</code>. 로드 시점에 r/z gate 용 <code>b_combined = b_ih + b_hh</code> 를 미리 계산(fused bias add 가 공유). n gate 는 reset gate <code>r</code> 이 hidden side 만 mask 하므로 <code>b_in</code> 과 <code>b_hn</code> 을 분리 유지.

### Compute pattern — full‑fused single kernel

RNNFormer block 당 GEMM 2개(<code>W_ih·x</code>, <code>W_hh·h</code>, 각 <code>[F2=72, 3D=216]</code> 출력)와 r/z/n gate update, h‑update 가 <b>per‑tier 단일 SIMD 커널 1개</b>에 모두 fuse 된다: <code>qgemm_&lt;tier&gt;_gru_full_fused_fp16inout</code> — 6개 tier 전부 구현(I8MM, DOTPROD, NEON, AVX‑512 VNNI, AVX‑VNNI, AVX2).

<table>
<thead><tr><th align="left">단계</th><th align="left">어디서</th></tr></thead>
<tbody>
<tr><td><code>W_ih·x</code></td><td>M‑block 당 작은 L1d‑resident <code>ih_tile</code> 에 안착 (int32 누산기는 메모리로 안 떨어짐)</td></tr>
<tr><td><code>W_hh·h</code></td><td>register 에서 바로 dequant</td></tr>
<tr><td>r/z/n gate</td><td>위 결과 + <code>ih_tile</code> + bsum 을 결합해 in‑place 생성</td></tr>
<tr><td>sigmoid / tanh</td><td>RNNoise rational polynomial + reciprocal-estimate; exact libm match 대상이 아니라 family 내 동일 계수·동일 reduction order 유지가 목표</td></tr>
</tbody>
</table>

<b>fp16‑inout:</b> 커널이 직전 hidden state 를 fp16 storage slot 에서 직접 읽고, <b>dual‑store ngate epilogue</b> 로 업데이트된 state 를 다시 쓴다 — 다음 <code>rnn_fc</code> 가 소비할 fp32 copy 하나는 엔진 스크래치로, fp16 copy 하나는 하드웨어 fp16 변환을 거쳐 <code>gru_h[block]</code> 으로. 이전 fp32‑inout path 가 필요로 했던 <b>GRU 앞 unpack + GRU 뒤 pack 패스를 둘 다 제거</b>한다. <sub>(전체 이력 → <a href="optimizations.md">optimizations.md</a>)</sub>

hidden state 버퍼 <code>gru_h[4][F2 × D] = [4][72 × 72]</code> 는 fp16 packed 이며 (<a href="#cross-stage-storage-fp16">아래</a>) 오디오 프레임 간 유지 + <code>fe_reset()</code> 으로만 reset.

### Per‑input dynamic quantization

두 GEMM 이 <b>각자 다른 per-frame activation scale</b>(<code>act_x</code> vs <code>act_h</code>)을 쓴다 — 입력 <code>x</code> 와 직전 hidden state <code>h</code> 의 magnitude 분포가 매우 달라서 scale 을 하나로 fuse 하면 작은 range 쪽이 clip 된다.

<details>
<summary><b>scalar fallback 부재</b> — dispatch contract</summary>

비‑fp16 scalar fp16‑cvt fallback path 는 빌드에 <b>존재하지 않는다</b>. 엔진은 fallback 을 돌리지 않고 <code>fe_init</code> 실패(non-zero return)로 빠진다 — dispatch contract 가 x86 측 AVX2+FMA3+F16C, aarch64 측 FCVTN/FCVTL 을 요구한다. fp16 변환이 mandatory 인 이유는 <a href="#cross-stage-storage-fp16">아래</a> 참고.
</details>

---

## Multi‑Head Self‑Attention

### 위치

4개 RNNFormer block 각각이 MHSA layer 1개씩. <b>attention 도 freq 축</b>("sequence" length = <code>F2 = 72</code> freq bin, 시간 축 아님). <b>시간 KV cache 없음</b> — 매 오디오 프레임이 자기 72 freq token 으로 attention 을 처음부터 재계산한다.

<table>
<thead><tr><th align="left">상수</th><th>값</th></tr></thead>
<tbody>
<tr><td>Sequence length N = F2</td><td>72</td></tr>
<tr><td>Model dim C = C2</td><td>72</td></tr>
<tr><td>Heads NH</td><td>4</td></tr>
<tr><td>Head dim HD = C / NH</td><td>18</td></tr>
<tr><td>QKV concat dim C3 = 3·C</td><td>216</td></tr>
</tbody>
</table>

### Stage 분해

Attention block 은 6개 fused stage. <b>softmax 만 fp32</b>, 나머지 matmul 은 모두 int8 W8A8.

<table>
<thead><tr><th>#</th><th align="left">Stage</th><th align="left">가중치</th><th align="left">양자화 / 비고</th></tr></thead>
<tbody>
<tr><td>1</td><td><b>QKV projection</b> — <code>[F2, C] · W_qkv<sup>T</sup></code></td><td><code>W_qkv [3C, C] = [216, 72]</code></td><td>출력 <code>[72, 216]</code>, per‑tensor <code>qkv_scale</code> 1개로 int8 in‑place 양자화. attention 의 유일한 weight 행렬(output FC 제외)</td></tr>
<tr><td>2</td><td><b>Q/K/V head‑major pack</b> (1‑pass)</td><td>—</td><td><code>[F2, 3C]</code> int8 버퍼 단일 패스로 Q(contiguous <code>[NH, F2, HD]</code>) + K(DOTPROD‑friendly scatter) + V<sup>T</sup>(transpose scatter) 동시 작성. I8MM tier 에서는 V pack 만 64-byte block layout 으로 즉시 repack 되고, Q@K<sup>T</sup> 는 아래 K20 DOTPROD 경로를 사용</td></tr>
<tr><td>3</td><td><b>Per‑head Q@K<sup>T</sup></b> — <code>[F2,HD]·[HD,F2]→[F2,F2]</code></td><td>(no weight)</td><td>int8 × int8 → int32. <code>1/√HD</code> 스케일은 per‑output <code>qkv_scale · qkv_scale / √HD</code> 에 fused. head 당 72²=5184 score, NH=4 → 프레임당 20.7 K</td></tr>
<tr><td>4</td><td><b>Row‑wise softmax + int8 quant</b></td><td>—</td><td>int32 직접 입력 → <code>combined_scale</code> 한 번에 dequant(전체 fp32 score matrix 없음; row scratch 만 사용) → softmax → int8, fixed symmetric softmax scale <code>1/127</code>. Schraudolph exp; reciprocal 은 NEON 에서 <code>vrecpe</code>+Newton, x86 에서는 scalar <code>1.0f/sum</code></td></tr>
<tr><td>5</td><td><b>Per‑head S@V<sup>T</sup></b> — <code>[F2,F2]·[F2,HD]→[F2,HD]</code></td><td>(no weight)</td><td>int8 × int8 → fp32. 각 head 가 <code>[F2, C]</code> concat 출력의 자기 슬롯에 씀</td></tr>
<tr><td>6</td><td><b>Output FC</b></td><td><code>[C, C] = [72, 72]</code></td><td>표준 W8A8 GEMM + bias, nonlinearity 없음. 출력이 residual stream 에 더해짐</td></tr>
</tbody>
</table>

### QKV weight 구조

fused projection weight <code>[3C, C] = [216, 72]</code> 는 행 축으로 slice 한다.

<table>
<thead><tr><th align="left">행 범위</th><th align="left">블록</th><th align="left">head h slice (<code>[18, 72]</code>)</th></tr></thead>
<tbody>
<tr><td><code>[0, C)</code></td><td>W_Q</td><td><code>W_qkv[h·HD : (h+1)·HD, :]</code></td></tr>
<tr><td><code>[C, 2C)</code></td><td>W_K</td><td><code>W_qkv[C + h·HD : C + (h+1)·HD, :]</code></td></tr>
<tr><td><code>[2C, 3C)</code></td><td>W_V</td><td><code>W_qkv[2C + h·HD : 2C + (h+1)·HD, :]</code></td></tr>
</tbody>
</table>

Output FC weight 은 <code>[C, C] = [72, 72]</code>.

### Activation interaction

- QKV projection 은 residual stream(post‑residual‑add)을 매 프레임 dynamic quantize 한다.
- projection output 은 fp32 scratch 에 한 번 materialize 해 maxabs 를 잡고 즉시 symmetric int8 로 재양자화한다. persistent fp32 Q/K/V 버퍼는 없다.
- 내부 Q@K<sup>T</sup> / S@V multiplications 는 <b>activation × activation</b>(weight 없음) → per‑tensor activation scale 만 필요 (Q·K·V 는 그 프레임의 <code>qkv_scale</code>, softmax 는 <code>1/127</code>).

---

## RNNFormer block flow

4개 block 각각이 다음 순서로 실행 (입력 = post‑rf_pre 스트림 <code>[F2, C2]</code>):

```
x = block_input                              # [F2, C2]
y = fe_gru_step(x, h_block)                  # [F2, D=C2]
y = rnn_fc(y) + x                            # FC [C2, C2] + bias + residual
                                             # rnn_fc.act 는 입력 range 를 진단용 기록
if block == 0:
    y = y + positional_embedding_table       # [F2, C2], block-0 만
attn = fe_mhsa(y) + y                        # MHSA + residual
block_output = attn                          # [F2, C2]
```

<table>
<thead><tr><th align="left">메모</th><th align="left">내용</th></tr></thead>
<tbody>
<tr><td><code>rnn_fc</code></td><td><code>[C2, C2]</code> int8 W8A8 Linear + bias</td></tr>
<tr><td>Residual</td><td><code>resA</code>(post‑FC) / <code>resB</code>(post‑attention) 는 fp32 add semantics. 현재 구현은 <code>rnn_fc</code> / <code>attn_fc</code> dequant epilogue 의 accumulating store(<code>C += ...</code>)로 fuse 한다</td></tr>
<tr><td>Positional embedding</td><td>block 0 만. <code>[F2, C2]</code> fp32 table 1회 로드 후 in‑place add</td></tr>
</tbody>
</table>

---

## 양자화 요약

전체 양자화 디자인(ISA‑specific 트릭, saturation 분석, static calibration freeze 실험의 phoneme‑blur 회귀)은 <a href="optimizations.md">optimizations.md</a>. 여기서는 모델 reference 에 필요한 계약만 요약한다.

<table>
<thead><tr><th align="left">대상</th><th align="left">방식</th></tr></thead>
<tbody>
<tr><td><b>Weights</b></td><td>int8 DOTPROD‑packed, per‑row symmetric scale (max‑abs/127). q8 blob 은 DOTPROD / AVX‑VNNI 친화 layout 으로 저장되고, I8MM tier 는 <code>fe_init</code> 로드 직후 in-place repack</td></tr>
<tr><td><b>Activations</b></td><td>per‑tensor asymmetric uint8, <b>매 프레임 재계산</b>(SIMD min/max scan → <code>(scale_x, zp_x)</code>, <code>int8 = u8 − 128</code> 으로 저장해 signed × signed 커널 재사용). zero‑point offset 은 dequant epilogue 의 <code>bias_eff</code> 항에 흡수. Static calibration freeze 는 없음; <code>FeActScale</code> 는 진단용 running min/max 만 기록</td></tr>
</tbody>
</table>

<b>bit‑identity:</b> 같은 arch family 안에서 ISA tier 간(ARM NEON ≡ DOTPROD ≡ I8MM, x86 AVX2 ≡ AVX‑VNNI ≡ AVX‑512 VNNI) 같은 입력 → bit‑identical 출력. fast‑math 없음, softmax / SiLU 의 tier 별 근사 없음. (SSE4.1 은 FMA3 부재로 이 보장이 깨져 x86 family 에서 drop.) 별개로, 현재 fp16-storage engine 은 historical fp32‑only engine 과 byte‑identical 이 <b>아니며</b>, 그 path 간 동등성은 STOI/SNR 로 assert 한다(SHA256 아님).

### 품질 reference

RNNoise demo clean target 대비 absolute quality (<code>tools/quality_compare.py</code>,
3 클립 babble/car/street @5 dB):

<div align="center">

<b>faster-enhancer.c q8 PESQ‑WB 1.837 vs clean target</b><br>
<sub>STOI 0.9442 · eSTOI 0.8726 · SNR 15.01 dB. fp32 ONNX 는 PESQ‑WB 1.833 · STOI 0.9440 · eSTOI 0.8721 · SNR 15.07 dB.</sub>

</div>

faster-enhancer.c q8 vs ONNX fp32 output fidelity:

<div align="center">

<b>PESQ‑WB 4.579 (ARM NEON/DOTPROD/I8MM byte‑id)</b><br>
<sub>STOI 0.9993 · eSTOI 0.9987 · SNR 34.04 dB · avg max|d| 0.0291.</sub>

</div>

fp32 reference 는 엔진과 <b>같은 causal STFT grid</b> 위에서 생성한다(<code>tools/onnx_reference_causal.py</code>) — <code>torch.stft</code> 기본값 <code>center=True</code> 는 analysis grid 를 192 샘플 어긋나게 해 fidelity 를 프레이밍 위상차로 오염시킨다(→ <a href="optimizations.md">optimizations.md</a>). 정합된 grid 에서 두 출력은 <b>lag 0</b> 으로 정렬된다. Cross‑tier byte‑id 는 <code>tools/sha256_matrix.sh</code> 로 검증한다.

---

## 레이어별 compute footprint

상수: F1 = 128, F2 = 72, C1 = 96, C2 = 72, NH = 4, HD = 18.

<table>
<thead><tr><th align="left">Layer</th><th align="left">Weight shape</th><th align="left">Dense-equivalent multiply footprint / frame</th></tr></thead>
<tbody>
<tr><td>enc_pre</td><td>[96, 16]</td><td>96 · 16 · F1</td></tr>
<tr><td>enc[0..2] (k=3 Winograd)</td><td>[96, 288] direct</td><td>3 · 4 · (F1/2) · 96 · 96</td></tr>
<tr><td>rf_pre_lin</td><td>[F2, F1]</td><td>F2 · F1 · C1</td></tr>
<tr><td>rf_pre_conv</td><td>[72, 96]</td><td>72 · 96 · F2</td></tr>
<tr><td>GRU W_ih (×4)</td><td>[216, 72]</td><td>4 · 216 · 72 · F2</td></tr>
<tr><td>GRU W_hh (×4)</td><td>[216, 72]</td><td>4 · 216 · 72 · F2</td></tr>
<tr><td>rnn_fc (×4)</td><td>[72, 72]</td><td>4 · 72 · 72 · F2</td></tr>
<tr><td>QKV proj (×4)</td><td>[216, 72]</td><td>4 · 216 · 72 · F2</td></tr>
<tr><td>Q@K<sup>T</sup> (×4)</td><td>(no weight)</td><td>4 · NH · F2 · HD · F2</td></tr>
<tr><td>S@V (×4)</td><td>(no weight)</td><td>4 · NH · F2 · HD · F2</td></tr>
<tr><td>attn_fc (×4)</td><td>[72, 72]</td><td>4 · 72 · 72 · F2</td></tr>
<tr><td>rf_post_lin</td><td>[F1, F2]</td><td>F1 · F2 · C2</td></tr>
<tr><td>rf_post_conv</td><td>[96, 72]</td><td>96 · 72 · F1</td></tr>
<tr><td>dec[0..2].1x1</td><td>[96, 192]</td><td>3 · 96 · 192 · F1</td></tr>
<tr><td>dec[0..2].3x3 Winograd</td><td>[96, 288] direct</td><td>3 · 4 · (F1/2) · 96 · 96</td></tr>
<tr><td>dec_post_1x1</td><td>[96, 192]</td><td>96 · 192 · F1</td></tr>
<tr><td>dec_post_up</td><td>[96, 16]</td><td>96 · 16 · F1 (fp32)</td></tr>
<tr><td>STFT + iSTFT</td><td>—</td><td>2 × 1024‑pt real FFT</td></tr>
</tbody>
</table>

<sub>k=3 rows 는 현재 런타임 Winograd F(2,3) inner GEMM multiply 수다. direct‑conv equivalent 로 세면 53.4 MMAC/frame 이지만, 현재 Winograd k=3 path 의 전체 dense-equivalent runtime multiply footprint 는 46.3 MMAC/frame (~6.95 GMAC/s @150 fps) 이다. 이 합계에는 fp32 <code>enc_pre</code> / <code>dec_post_up</code> edge conv 도 포함된다. <code>dec_post_up</code> 의 boundary guard 가 실제 루프에서 768 multiply-add 를 skip 하지만 표시 precision 에서는 46.3 MMAC 그대로다. Multiply footprint 는 <b>tier‑불변</b>이지만 처리 폭은 dispatch 된 tier 마다 다르다(AVX‑512 zmm 16‑lane vs 다른 tier ymm 8‑lane). 따라서 GRU·activation‑quantize 의 절대 비중은 AVX‑512 에서 더 작다. 런타임 stage share 는 CPU/compiler/tier/profile instrumentation 여부에 따라 달라지므로 <a href="optimizations.md#current-profile-map">profile build</a> 에서 다시 뽑는다.</sub>

---

## Cross‑stage storage (fp16)

레이어 간 long‑lived state 버퍼는 모두 IEEE binary16(fp16) 으로 저장한다 — <b>mandatory storage</b>. current <code>sizeof(FeState)=432,384</code> bytes, named state/staging delta 는 188,928 bytes (184.5 KiB). 전체 stage 이력은 <a href="optimizations.md">optimizations.md</a>.

<table>
<thead><tr><th align="left">버퍼</th><th align="left">Shape</th><th>Storage</th><th align="left">용도</th></tr></thead>
<tbody>
<tr><td><code>enc_skip</code></td><td>[ENC_BLOCKS+1][C1·F1] = [4][96·128]</td><td>fp16</td><td>encoder feature snapshot, decoder 에 concat</td></tr>
<tr><td><code>gru_h</code></td><td>[RF_BLOCKS][F2·C2] = [4][72·72]</td><td>fp16</td><td>recurrent hidden state, 오디오 프레임 간 유지</td></tr>
</tbody>
</table>

<b>하드웨어 요구사항:</b> fp16 변환은 모든 지원 ISA 에서 mandatory — x86 최저 tier 는 AVX2+FMA3+F16C(VCVTPS2PH / VCVTPH2PS), aarch64 NEON 은 FCVTN / FCVTL. scalar fp16 fallback 은 없다 — 있으면 소리없이 안 돌아갈 위험이라 미지원 host 는 <code>fe_init</code> 실패로 처리한다.

<b>커널 통합:</b> GRU 는 dual‑store ngate epilogue 로 fp32(즉시 소비자 <code>rnn_fc</code>) + fp16(다음 프레임 <code>gru_h</code>) 동시 write. Encoder skip 은 <code>fe_silu_skip_fp16</code> 가 forward 중 <code>enc_skip</code> 에 fp16 직접 write, decoder concat‑1x1 이 fp16 직접 read → fused transposed‑input quantize 에서 int8 emit. <code>attn_*</code> 스크래치는 intra‑frame(매 호출 zero‑re‑init)이라 fp16 화 이득이 없어 제외.

---

## 최적화 관련 제약 / 속성

- <b>Single‑threaded inference</b>, per audio frame, CPU only. 타깃: macOS · iOS · Android · Windows(GNU-style Clang/MinGW 계열, MSVC/clang-cl 제외). ARM tier NEON / DOTPROD / I8MM, x86 tier AVX2 / AVX‑VNNI / AVX‑512 VNNI (AVX2+FMA3+F16C 최저, SSE4.1 drop).
- <b>Fine‑tuning / training 인프라 없음.</b> 모든 최적화는 post‑training (학습된 weight 에 transformation 적용 → q8 export → run).
- <b>Streaming:</b> 매 프레임 320 sample = 6.67 ms. STFT/iSTFT 가 1024‑sample window overlap‑add, 엔진이 필요한 cache 유지. GRU hidden state 와 몇몇 작은 per‑block streaming 버퍼가 프레임 간 유지.
- <b>Q/K ~ softmax 사이 nonlinearity 없음</b> (<code>1/√HD</code> scale 제외).
- <b>Frequency‑axis attention:</b> sequence length 가 72 로 고정, 시간에 따라 자라지 않음. autoregressive KV cache 없음 — 매 프레임이 head 당 <code>[72, 72]</code> score matrix 새로 계산.
- <b>Production GEMM 의 M/N tile 차원은 8 의 배수</b> — scalar remainder kernel 이 필요 없도록 compile-time assert 로 고정한다. K 축은 tier 별 계약을 따른다: DOTPROD 는 k4, I8MM 은 k8 이며, attention Q@K 의 K20 은 I8MM 전용 kernel 없이 DOTPROD K20 route 를 쓴다.
- <b>모델 511,754 params, 565,108-byte (551.9 KiB) q8 blob.</b> 10 % 0‑shot magnitude pruning 만 걸어도 PESQ 1.66 → 1.19 으로 실패 — 대부분 weight 가 "load‑bearing".

---

## 코드 위치

<table>
<thead><tr><th align="left">영역</th><th align="left">파일</th></tr></thead>
<tbody>
<tr><td>Engine main loop</td><td><a href="../src/fe_engine.c"><code>src/fe_engine.c</code></a></td></tr>
<tr><td>Conv1d entry wrappers (k=1 + Winograd k=3)</td><td><a href="../src/nn/fe_conv.c"><code>src/nn/fe_conv.c</code></a></td></tr>
<tr><td>Winograd F(2,3)</td><td><a href="../src/winograd/fe_winograd.c"><code>src/winograd/fe_winograd.c</code></a></td></tr>
<tr><td>FFT dispatch</td><td><a href="../src/fe_fft.c"><code>src/fe_fft.c</code></a></td></tr>
<tr><td>Per‑tier FFT kernels</td><td><a href="../src/fft/"><code>src/fft/fft_*.c</code></a></td></tr>
<tr><td>STFT / iSTFT</td><td><a href="../src/fe_stft.c"><code>src/fe_stft.c</code></a></td></tr>
<tr><td>MHSA stages</td><td><a href="../src/nn/fe_attention.c"><code>src/nn/fe_attention.c</code></a></td></tr>
<tr><td>GRU step + dispatch</td><td><a href="../src/nn/fe_gru.c"><code>src/nn/fe_gru.c</code></a></td></tr>
<tr><td>Activations + softmax</td><td><a href="../src/nn/fe_activations.c"><code>src/nn/fe_activations.c</code></a></td></tr>
<tr><td>GRU full‑fused kernels</td><td><a href="../src/qgemm/"><code>src/qgemm/{arm,x86}/</code></a></td></tr>
</tbody>
</table>
