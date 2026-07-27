# FastEnhancer‑Medium 48 kHz — layer reference

<b>Single self‑contained description of the Conv / GRU / Attention layers as deployed in the engine.</b><br>
<sub>Authoritative source for "what the model is" — for "how it was optimized" see <a href="../README.md">README</a> + <a href="optimizations.md">optimizations.md</a>.</sub>

> <b>TL;DR</b> — 48 kHz monaural denoiser. STFT → encoder (×4 downsample on the F axis) → RNNFormer × 4 → decoder (×4 upsample) → iSTFT. RNNFormer block = <code>GRU → rnn_fc(+res) → MHSA(+res)</code>, all operating on the <b>frequency axis</b>. W8A8 int8, 511,754 params, 320-sample processing quantum (6.67 ms), zero look‑ahead, fixed STFT alignment delay 704 samples (14.67 ms).

---

## Model at a glance

<table>
<thead><tr><th align="left">Item</th><th align="left">Value</th></tr></thead>
<tbody>
<tr><td>Task</td><td>real‑time monaural noise suppression</td></tr>
<tr><td>Sample rate</td><td>48 kHz</td></tr>
<tr><td>Frame / hop</td><td>320 samples = 6.67 ms</td></tr>
<tr><td>STFT</td><td>n_fft = 1024, overlap‑add kept across <code>fe_run()</code> calls</td></tr>
<tr><td>Latency / alignment</td><td>zero look‑ahead; fixed STFT alignment delay 704 samples (14.67 ms), one call emits 320 samples</td></tr>
<tr><td>Params</td><td>511,754 (565,108-byte / 551.9 KiB q8 blob; SHA-256 <code>d6f98e186bee6ed655b52a2edf5f707ccf761c755bb69de8de7f68cad42d7837</code>; 2,047,016-byte / 1.95 MiB fp32-equivalent)</td></tr>
<tr><td>Working tensors</td><td><code>[freq, channels]</code> row‑major throughout</td></tr>
<tr><td>Channels</td><td>encoder / decoder C1 = 96 · RNNFormer C2 = 72</td></tr>
<tr><td>Freq dims</td><td>F1 = 128 after enc_pre · F2 = 72 inside RNNFormer</td></tr>
<tr><td>Heads</td><td>NH = 4, HD = C2 / NH = 18</td></tr>
<tr><td>Nonlinearity</td><td>SiLU only (attention softmax and GRU sigmoid/tanh are separate)</td></tr>
</tbody>
</table>

<div align="center">

<code>audio → STFT → encoder (F axis stride 4 downsample) → RNNFormer × 4 → decoder (F axis stride 4 upsample) → iSTFT → audio</code>

</div>

<b>Input compression:</b> magnitude is compressed by <code>^0.3</code> before entering the network and inverted with <code>^(1/0.3)</code> on output. Phase passes through untouched — the network emits only a magnitude mask, multiplied onto the input phase.

<b>Quantization:</b> W8A8 — per‑row symmetric int8 weights, per‑tensor asymmetric uint8 activations <b>recomputed every frame</b> (dynamic quantize). Summary <a href="#quantization-summary">below</a>, full design in <a href="optimizations.md">optimizations.md</a>.

<b>Cross‑stage state:</b> recurrent and skip buffers are both stored fp16‑packed (<a href="#cross-stage-storage-fp16">below</a>).

---

## Convolutions

Every convolution is a 1‑D conv applied along the <b>frequency axis</b> (not the time axis). Input <code>[freq=F, Cin]</code> → output <code>[freq=F, Cout]</code>.

### Layer inventory

<table>
<thead><tr>
<th align="left">Layer</th><th>Kernel</th><th>In ch</th><th>Out ch</th>
<th align="left">Weight (Cout × Cin × k)</th><th align="left">Path</th><th>Quantization</th>
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

The 3 encoder blocks are wired to the 3 matching decoder blocks by concat‑style skip connections — the encoder output is concatenated on the channel axis with the upsampled decoder feature (channel count 2·C1), then <code>dec[i].1x1</code> mixes it back down to C1.

### k=3 conv — Winograd F(2,3)

The k=3 convs in encoder / decoder (Cin=Cout=96) run as <b>Winograd F(2,3)</b>, cutting MACs by <b>−33 %</b> versus direct im2col.

<table>
<thead><tr><th align="left">Item</th><th align="left">Detail</th></tr></thead>
<tbody>
<tr><td>Pass structure</td><td><b>Pass A</b> (vmax‑only, no storage) + <b>Pass B</b> (re‑read + per‑component scale + int8 quantize)</td></tr>
<tr><td>Inner GEMM</td><td>4× <code>gemm_int32</code> (dequant epilogue skipped)</td></tr>
<tr><td>Epilogue</td><td>batched fused — 4× int32 U → dequant + A<sup>T</sup> (even sum, odd signed sum) + bias + optional SiLU in a single pass</td></tr>
</tbody>
</table>

The output is fp32; the next layer's quantize pass re‑quantizes it with dynamic min/max. Weights are logically <code>[Cout, Cin, k]</code> but stored as packed int8 plus one fp32 scale per output row. The production q8 loader derives the Winograd weights for every k=3 layer right after load, and treats a failure in that step as a load failure.

### k=1 conv

A k=1 conv is a per‑freq linear layer. Input <code>[F, Cin]</code> → output <code>[F, Cout]</code>, weight <code>[Cout, Cin]</code> — a single W8A8 GEMM + bias + optional SiLU + optional int8 output. The decoder concat 1×1 projects a <code>[F, 2·C1]</code> tile (encoder skip and current decoder feature concatenated on the channel axis) down to <code>[F, C1]</code>.

### fp32‑only layers

<table>
<thead><tr><th align="left">Layer</th><th align="left">Reason</th></tr></thead>
<tbody>
<tr><td><b>enc_pre</b></td><td>The StridedConv reshape trick rearranges the 2‑channel (re, im) input into Ci'=8 channels at 1/4 freq resolution → k=2 conv. Kept fp32 because of the input distribution.</td></tr>
<tr><td><b>dec_post_up</b></td><td>The final ConvTranspose1d (kernel 8, stride 4) upsamples to <code>[2, F=512]</code>. Kept fp32.</td></tr>
</tbody>
</table>

---

## GRU

### Placement

Each of the 4 RNNFormer blocks contains one GRU step. Applied along the <b>frequency axis</b> — each freq bin acts as a "timestep" of the scan, and the hidden state is held per freq bin and <b>carried across audio frames</b>.

### Shape & equations

- Hidden dim <b>D = 72</b> (= C2).
- Gates: standard PyTorch <code>[r, z, n]</code>, linear‑before‑reset (<code>r</code> masks the hidden‑side <code>n</code>‑gate input).

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

Weights are stored as <b>two fused matrices</b>.

<table>
<thead><tr><th align="left">Matrix</th><th align="left">Shape</th><th align="left">Row order</th></tr></thead>
<tbody>
<tr><td><code>W_ih</code> (input‑side)</td><td>[3D, D] = [216, 72]</td><td><code>[W_ir; W_iz; W_in]</code></td></tr>
<tr><td><code>W_hh</code> (hidden‑side)</td><td>[3D, D] = [216, 72]</td><td>same</td></tr>
</tbody>
</table>

Biases <code>b_ih</code>, <code>b_hh</code> are <code>[3D]</code> each. At load time <code>b_combined = b_ih + b_hh</code> is precomputed for the r/z gates (shared by the fused bias add). The n gate keeps <code>b_in</code> and <code>b_hn</code> separate, because the reset gate <code>r</code> masks only the hidden side.

### Compute pattern — full‑fused single kernel

Per RNNFormer block, the 2 GEMMs (<code>W_ih·x</code>, <code>W_hh·h</code>, each producing <code>[F2=72, 3D=216]</code>) plus the r/z/n gate update and the h‑update are all fused into <b>one single SIMD kernel per tier</b>: <code>qgemm_&lt;tier&gt;_gru_full_fused_fp16inout</code> — implemented for all 6 tiers (I8MM, DOTPROD, NEON, AVX‑512 VNNI, AVX‑VNNI, AVX2).

<table>
<thead><tr><th align="left">Step</th><th align="left">Where</th></tr></thead>
<tbody>
<tr><td><code>W_ih·x</code></td><td>lands in a small L1d‑resident <code>ih_tile</code> per M‑block (int32 accumulators never spill to memory)</td></tr>
<tr><td><code>W_hh·h</code></td><td>dequantized straight out of registers</td></tr>
<tr><td>r/z/n gate</td><td>produced in place by combining the above with <code>ih_tile</code> and bsum</td></tr>
<tr><td>sigmoid / tanh</td><td>RNNoise rational polynomial + reciprocal-estimate; not aiming at an exact libm match — the goal is identical coefficients and identical reduction order within a family</td></tr>
</tbody>
</table>

<b>fp16‑inout:</b> the kernel reads the previous hidden state directly from the fp16 storage slot and writes the updated state back through a <b>dual‑store ngate epilogue</b> — one fp32 copy into engine scratch for the following <code>rnn_fc</code> to consume, one fp16 copy through the hardware fp16 conversion into <code>gru_h[block]</code>. This <b>removes both the unpack pass before the GRU and the pack pass after it</b> that the earlier fp32‑inout path needed. <sub>(full history → <a href="optimizations.md">optimizations.md</a>)</sub>

The hidden state buffer <code>gru_h[4][F2 × D] = [4][72 × 72]</code> is fp16 packed (<a href="#cross-stage-storage-fp16">below</a>), carried across audio frames and reset only by <code>fe_reset()</code>.

### Per‑input dynamic quantization

The two GEMMs use <b>separate per-frame activation scales</b> (<code>act_x</code> vs <code>act_h</code>) — the magnitude distributions of the input <code>x</code> and the previous hidden state <code>h</code> differ enough that fusing them into one scale clips whichever side has the smaller range.

<details>
<summary><b>No scalar fallback</b> — dispatch contract</summary>

A non‑fp16 scalar fp16‑cvt fallback path <b>does not exist</b> in the build. Rather than running a fallback, the engine fails <code>fe_init</code> (non-zero return) — the dispatch contract requires AVX2+FMA3+F16C on x86 and FCVTN/FCVTL on aarch64. Why fp16 conversion is mandatory: see <a href="#cross-stage-storage-fp16">below</a>.
</details>

---

## Multi‑Head Self‑Attention

### Placement

Each of the 4 RNNFormer blocks has one MHSA layer. <b>Attention runs on the freq axis too</b> ("sequence" length = <code>F2 = 72</code> freq bins, not time). <b>No temporal KV cache</b> — every audio frame recomputes attention from scratch over its own 72 freq tokens.

<table>
<thead><tr><th align="left">Constant</th><th>Value</th></tr></thead>
<tbody>
<tr><td>Sequence length N = F2</td><td>72</td></tr>
<tr><td>Model dim C = C2</td><td>72</td></tr>
<tr><td>Heads NH</td><td>4</td></tr>
<tr><td>Head dim HD = C / NH</td><td>18</td></tr>
<tr><td>QKV concat dim C3 = 3·C</td><td>216</td></tr>
</tbody>
</table>

### Stage breakdown

The attention block is 6 fused stages. <b>Only softmax is fp32</b>; every other matmul is int8 W8A8.

<table>
<thead><tr><th>#</th><th align="left">Stage</th><th align="left">Weights</th><th align="left">Quantization / notes</th></tr></thead>
<tbody>
<tr><td>1</td><td><b>QKV projection</b> — <code>[F2, C] · W_qkv<sup>T</sup></code></td><td><code>W_qkv [3C, C] = [216, 72]</code></td><td>output <code>[72, 216]</code>, quantized in place to int8 with a single per‑tensor <code>qkv_scale</code>. The only weight matrix in attention (besides the output FC)</td></tr>
<tr><td>2</td><td><b>Q/K/V head‑major pack</b> (1‑pass)</td><td>—</td><td>a single pass over the <code>[F2, 3C]</code> int8 buffer writes Q (contiguous <code>[NH, F2, HD]</code>), K (DOTPROD‑friendly scatter) and V<sup>T</sup> (transpose scatter) at once. On the I8MM tier only the V pack is immediately repacked into a 64-byte block layout; Q@K<sup>T</sup> uses the K20 DOTPROD route described below</td></tr>
<tr><td>3</td><td><b>Per‑head Q@K<sup>T</sup></b> — <code>[F2,HD]·[HD,F2]→[F2,F2]</code></td><td>(no weight)</td><td>int8 × int8 → int32. The <code>1/√HD</code> scale is fused into the per‑output <code>qkv_scale · qkv_scale / √HD</code>. 72²=5184 scores per head, NH=4 → 20.7 K per frame</td></tr>
<tr><td>4</td><td><b>Row‑wise softmax + int8 quant</b></td><td>—</td><td>int32 straight in → dequant by <code>combined_scale</code> in one step (no full fp32 score matrix; row scratch only) → softmax → int8, fixed symmetric softmax scale <code>1/127</code>. Schraudolph exp; the reciprocal is <code>vrecpe</code>+Newton on NEON and scalar <code>1.0f/sum</code> on x86</td></tr>
<tr><td>5</td><td><b>Per‑head S@V<sup>T</sup></b> — <code>[F2,F2]·[F2,HD]→[F2,HD]</code></td><td>(no weight)</td><td>int8 × int8 → fp32. Each head writes into its own slot of the <code>[F2, C]</code> concat output</td></tr>
<tr><td>6</td><td><b>Output FC</b></td><td><code>[C, C] = [72, 72]</code></td><td>standard W8A8 GEMM + bias, no nonlinearity. The output is added into the residual stream</td></tr>
</tbody>
</table>

### QKV weight layout

The fused projection weight <code>[3C, C] = [216, 72]</code> is sliced along the row axis.

<table>
<thead><tr><th align="left">Row range</th><th align="left">Block</th><th align="left">head h slice (<code>[18, 72]</code>)</th></tr></thead>
<tbody>
<tr><td><code>[0, C)</code></td><td>W_Q</td><td><code>W_qkv[h·HD : (h+1)·HD, :]</code></td></tr>
<tr><td><code>[C, 2C)</code></td><td>W_K</td><td><code>W_qkv[C + h·HD : C + (h+1)·HD, :]</code></td></tr>
<tr><td><code>[2C, 3C)</code></td><td>W_V</td><td><code>W_qkv[2C + h·HD : 2C + (h+1)·HD, :]</code></td></tr>
</tbody>
</table>

The output FC weight is <code>[C, C] = [72, 72]</code>.

### Activation interaction

- The QKV projection dynamically quantizes the residual stream (post‑residual‑add) every frame.
- The projection output is materialized once into fp32 scratch to take its maxabs, then immediately re‑quantized to symmetric int8. There are no persistent fp32 Q/K/V buffers.
- The internal Q@K<sup>T</sup> / S@V multiplications are <b>activation × activation</b> (no weights) → only per‑tensor activation scales are needed (that frame's <code>qkv_scale</code> for Q·K·V, <code>1/127</code> for softmax).

---

## RNNFormer block flow

Each of the 4 blocks runs in this order (input = the post‑rf_pre stream <code>[F2, C2]</code>):

```
x = block_input                              # [F2, C2]
y = fe_gru_step(x, h_block)                  # [F2, D=C2]
y = rnn_fc(y) + x                            # FC [C2, C2] + bias + residual
                                             # rnn_fc.act records the input range for diagnostics
if block == 0:
    y = y + positional_embedding_table       # [F2, C2], block-0 only
attn = fe_mhsa(y) + y                        # MHSA + residual
block_output = attn                          # [F2, C2]
```

<table>
<thead><tr><th align="left">Note</th><th align="left">Detail</th></tr></thead>
<tbody>
<tr><td><code>rnn_fc</code></td><td><code>[C2, C2]</code> int8 W8A8 Linear + bias</td></tr>
<tr><td>Residual</td><td><code>resA</code> (post‑FC) / <code>resB</code> (post‑attention) have fp32 add semantics. The current implementation fuses them into the accumulating store (<code>C += ...</code>) of the <code>rnn_fc</code> / <code>attn_fc</code> dequant epilogue</td></tr>
<tr><td>Positional embedding</td><td>block 0 only. <code>[F2, C2]</code> fp32 table loaded once, added in place</td></tr>
</tbody>
</table>

---

## Quantization summary

The full quantization design (ISA‑specific tricks, saturation analysis, the phoneme‑blur regression from the static calibration freeze experiment) is in <a href="optimizations.md">optimizations.md</a>. Only the contract needed for the model reference is summarized here.

<table>
<thead><tr><th align="left">Target</th><th align="left">Scheme</th></tr></thead>
<tbody>
<tr><td><b>Weights</b></td><td>int8 DOTPROD‑packed, per‑row symmetric scale (max‑abs/127). The q8 blob is stored in a DOTPROD / AVX‑VNNI friendly layout; the I8MM tier repacks it in place right after the <code>fe_init</code> load</td></tr>
<tr><td><b>Activations</b></td><td>per‑tensor asymmetric uint8, <b>recomputed every frame</b> (SIMD min/max scan → <code>(scale_x, zp_x)</code>, stored as <code>int8 = u8 − 128</code> so the signed × signed kernels are reused). The zero‑point offset is absorbed into the <code>bias_eff</code> term of the dequant epilogue. No static calibration freeze; <code>FeActScale</code> only records running min/max for diagnostics</td></tr>
</tbody>
</table>

<b>bit‑identity:</b> within the same arch family, ISA tiers (ARM NEON ≡ DOTPROD ≡ I8MM, x86 AVX2 ≡ AVX‑VNNI ≡ AVX‑512 VNNI) produce bit‑identical output for the same input. No fast‑math, no per‑tier approximations of softmax / SiLU. (SSE4.1 breaks this guarantee for lack of FMA3 and was dropped from the x86 family.) Separately, the current fp16-storage engine is <b>not</b> byte‑identical to the historical fp32‑only engine; equivalence between those paths is asserted via STOI/SNR, not SHA256.

### Quality reference

Absolute quality against the RNNoise demo clean target (<code>tools/quality_compare.py</code>,
3 clips babble/car/street @5 dB):

<div align="center">

<b>faster-enhancer.c q8 PESQ‑WB 1.837 vs clean target</b><br>
<sub>STOI 0.9442 · eSTOI 0.8726 · SNR 15.01 dB. fp32 ONNX gives PESQ‑WB 1.833 · STOI 0.9440 · eSTOI 0.8721 · SNR 15.07 dB.</sub>

</div>

faster-enhancer.c q8 vs ONNX fp32 output fidelity:

<div align="center">

<b>PESQ‑WB 4.579 (ARM NEON/DOTPROD/I8MM byte‑id)</b><br>
<sub>STOI 0.9993 · eSTOI 0.9987 · SNR 34.04 dB · avg max|d| 0.0291.</sub>

</div>

The fp32 reference is generated on the <b>same causal STFT grid</b> as the engine (<code>tools/onnx_reference_causal.py</code>) — the <code>torch.stft</code> default <code>center=True</code> shifts the analysis grid by 192 samples and contaminates fidelity with a framing phase offset (→ <a href="optimizations.md">optimizations.md</a>). On the matched grid the two outputs align at <b>lag 0</b>. Cross‑tier byte‑id is verified with <code>tools/sha256_matrix.sh</code>.

---

## Per‑layer compute footprint

Constants: F1 = 128, F2 = 72, C1 = 96, C2 = 72, NH = 4, HD = 18.

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

<sub>The k=3 rows are the multiply counts of the Winograd F(2,3) inner GEMM as it runs today. Counted as a direct‑conv equivalent it is 53.4 MMAC/frame, but the total dense-equivalent runtime multiply footprint of the current Winograd k=3 path is 46.3 MMAC/frame (~6.95 GMAC/s @150 fps). That total includes the fp32 <code>enc_pre</code> / <code>dec_post_up</code> edge convs. The boundary guard in <code>dec_post_up</code> skips 768 multiply-adds in the actual loop, but at the displayed precision the figure stays 46.3 MMAC. The multiply footprint is <b>tier‑invariant</b>, but the processing width differs per dispatched tier (AVX‑512 zmm 16‑lane vs ymm 8‑lane on the other tiers). So the absolute share of GRU and activation‑quantize is smaller on AVX‑512. Runtime stage share varies with CPU/compiler/tier and whether profile instrumentation is on, so take it again from a <a href="optimizations.md#current-profile-map">profile build</a>.</sub>

---

## Cross-stage storage (fp16)

All long‑lived state buffers between layers are stored as IEEE binary16 (fp16) — <b>mandatory storage</b>. Current <code>sizeof(FeState)=432,384</code> bytes, named state/staging delta is 188,928 bytes (184.5 KiB). Full stage history in <a href="optimizations.md">optimizations.md</a>.

<table>
<thead><tr><th align="left">Buffer</th><th align="left">Shape</th><th>Storage</th><th align="left">Use</th></tr></thead>
<tbody>
<tr><td><code>enc_skip</code></td><td>[ENC_BLOCKS+1][C1·F1] = [4][96·128]</td><td>fp16</td><td>encoder feature snapshot, concatenated into the decoder</td></tr>
<tr><td><code>gru_h</code></td><td>[RF_BLOCKS][F2·C2] = [4][72·72]</td><td>fp16</td><td>recurrent hidden state, carried across audio frames</td></tr>
</tbody>
</table>

<b>Hardware requirement:</b> fp16 conversion is mandatory on every supported ISA — the lowest x86 tier is AVX2+FMA3+F16C (VCVTPS2PH / VCVTPH2PS), aarch64 NEON uses FCVTN / FCVTL. There is no scalar fp16 fallback — having one would risk failing silently, so an unsupported host is handled as an <code>fe_init</code> failure.

<b>Kernel integration:</b> the GRU writes fp32 (for the immediate consumer <code>rnn_fc</code>) and fp16 (for the next frame's <code>gru_h</code>) at once via the dual‑store ngate epilogue. For the encoder skip, <code>fe_silu_skip_fp16</code> writes fp16 directly into <code>enc_skip</code> during the forward pass, and the decoder concat‑1x1 reads that fp16 directly, emitting int8 from the fused transposed‑input quantize. The <code>attn_*</code> scratch is intra‑frame (zero‑re‑init on every call), so there is nothing to gain from fp16 and it is excluded.

---

## Optimization‑relevant constraints / properties

- <b>Single‑threaded inference</b>, per audio frame, CPU only. Targets: macOS · iOS · Android · Windows (GNU-style Clang/MinGW family, excluding MSVC/clang-cl). ARM tiers NEON / DOTPROD / I8MM, x86 tiers AVX2 / AVX‑VNNI / AVX‑512 VNNI (AVX2+FMA3+F16C is the floor, SSE4.1 dropped).
- <b>No fine‑tuning / training infrastructure.</b> Every optimization is post‑training (apply a transformation to trained weights → q8 export → run).
- <b>Streaming:</b> 320 samples = 6.67 ms per frame. STFT/iSTFT does overlap‑add on a 1024‑sample window and the engine keeps the required caches. The GRU hidden state and a few small per‑block streaming buffers persist across frames.
- <b>No nonlinearity between Q/K and softmax</b> (other than the <code>1/√HD</code> scale).
- <b>Frequency‑axis attention:</b> sequence length is fixed at 72 and does not grow with time. No autoregressive KV cache — every frame computes a fresh <code>[72, 72]</code> score matrix per head.
- <b>The M/N tile dimensions of the production GEMM are multiples of 8</b> — pinned by compile-time asserts so that no scalar remainder kernel is needed. The K axis follows the per‑tier contract: k4 for DOTPROD, k8 for I8MM, and the K20 of attention Q@K uses the DOTPROD K20 route instead of an I8MM‑specific kernel.
- <b>511,754 params, 565,108-byte (551.9 KiB) q8 blob.</b> Even 10 % 0‑shot magnitude pruning fails at PESQ 1.66 → 1.19 — most weights are "load‑bearing".

---

## Code locations

<table>
<thead><tr><th align="left">Area</th><th align="left">File</th></tr></thead>
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
