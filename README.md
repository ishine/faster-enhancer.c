# faster-enhancer.c

A dependency-free C/SIMD int8 runtime for **FastEnhancer-Medium** at 48 kHz —
streaming speech enhancement that runs in a fraction of one CPU core, with no
inference framework, no heap allocation after startup, and no retraining.

It runs the [published FastEnhancer weights](https://github.com/aask1357/fastenhancer)
unchanged. The architecture is untouched; only the runtime is specialized.

```
0.069 real-time factor   on one Apple M2 core      (I8MM, racing)
0.096                    on a Galaxy S23+          (Snapdragon 8 Gen 2)
565,108 bytes            W8A8 weight blob
162 KiB                  static library, macOS arm64
```

---

## What it is

One fixed model, six hand-written int8 GEMM kernel tiers, one of which is
selected at initialization. There is no scalar fallback: a host that does not
meet the baseline ISA fails at `fe_init` rather than silently taking a slow
path.

- **Training-free.** Post-training quantization only. No QAT and no
  calibration set — activation ranges are recomputed from each frame.
- **Streaming and causal.** 320 samples in, 320 out, no look-ahead. The STFT
  contributes a fixed 704-sample (14.67 ms) alignment delay.
- **Allocation-free after init.** All state lives in one 432,384-byte
  structure, so a long run cannot drift into allocator jitter.
- **Byte-identical across tiers.** Within an architecture family, NEON,
  DOTPROD and I8MM produce bit-for-bit identical output, enforced by a
  regression gate. The same holds for the x86 tiers.

## Quick start

The public API is four functions.

```c
#include "fe.h"

int  fe_init (const void *weights_blob, int weights_size);
void fe_run  (const float *in, float *out);   /* 320 in -> 320 out */
void fe_reset(void);
void fe_free (void);
```

A minimal streaming loop:

```c
if (fe_init(weights, weights_size) != 0) return -1;
while (read_320_samples(in_buf)) {
    fe_run(in_buf, out_buf);   /* in_buf == out_buf is fine (in-place) */
    write_320_samples(out_buf);
}
fe_free();
```

## Build

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

No external dependencies. Produces `libfe.a` plus the runners in `build/`.

To try it on real audio, fetch the test clips first (needs `curl` and
`ffmpeg`; the clips are downloaded from their original location rather than
redistributed here):

```sh
./testaudio/fetch.sh
./build/fe_run_file weights/fe.q8 testaudio/street_5dB.f32 out.wav 5569 i8mm
```

`fe_run_file` reports per-frame percentiles against the 6.667 ms frame budget.
Two environment knobs are useful: `FE_QOS=hi|lo|off` picks a QoS class, and
`FE_PACE=1` feeds one frame per frame period instead of racing.

## Performance

Measured with 5569-frame clips, steady-state percentiles after discarding the
first 200 frames, median over repeats.

| Device | Tier | p50 RTF | p99 RTF | ms/frame |
|---|---|---:|---:|---:|
| Apple M2 | I8MM | **0.069** | 0.084 | 0.458 |
| Apple M2 | DOTPROD | 0.068 | 0.081 | 0.452 |
| Apple M2 | NEON | 0.163 | 0.187 | 1.085 |
| Galaxy S23+ | I8MM | **0.096** | 0.105 | 0.640 |
| Galaxy S23+ | DOTPROD | 0.113 | 0.122 | 0.753 |
| Galaxy S23+ | NEON | 0.344 | 0.365 | 2.293 |

Two things worth knowing before reading those as deployment cost.

**A benchmark races; an audio callback does not.** Feeding frames as fast as
the core accepts them holds the clock at maximum. A real callback delivers one
frame per 6.667 ms and lets the core idle, and the governor responds. Paced on
an M2 P-core the same work costs 4.2x more per frame (0.286 RTF) while using
49% less energy. Both are real-time; they answer different questions.

**The x86 tiers are not timed here.** They build and pass the tier-equality
gate, but this table is ARM only.

## Quality

The int8 engine is compared against the fp32 ONNX graph of the same weights,
on the engine's own causal analysis grid. Scoring a streaming engine against a
default centered-STFT reference measures framing phase rather than
quantization error: the two grids differ by 192 samples, which is not a
multiple of the 320-sample hop, so the offset does not cancel.

Over the 824-utterance VoiceBank-DEMAND test set at its native 48 kHz:

| | PESQ | STOI | SNR | LSD |
|---|---:|---:|---:|---:|
| noisy input | 1.967 | 0.9211 | 8.39 | 14.72 |
| fp32 ONNX | 3.060 | 0.9512 | 19.43 | 12.56 |
| **fe q8** | 3.054 | 0.9509 | 19.35 | 12.33 |

The port tracks the fp32 model to -0.006 PESQ and -0.08 dB SNR. Quantization
costs about 1.6% of what the enhancement itself gains.

## Supported targets

| ISA | Tier | Instruction | Requires |
|---|---|---|---|
| arm64 | NEON | `vmull_s8` -> `vmlal_s8` -> `vpadalq_s16` | baseline |
| arm64 | DOTPROD | `vdotq_laneq_s32` | FEAT_DotProd |
| arm64 | I8MM | `vmmlaq_s32` | FEAT_I8MM |
| x86-64 | AVX2 | `vpmovsxbw` + `vpmaddwd` | AVX2 + FMA3 + F16C |
| x86-64 | AVX-VNNI | `vpdpbusd ymm` | Alder Lake+, Zen 4 |
| x86-64 | AVX-512 VNNI | `vpdpbusd zmm` | Ice Lake / SPR, Zen 4 |

SSE4.1 is deliberately unsupported: without FMA3 the dequantization epilogue
becomes a two-step rounding that drifts until bit-identity breaks.

The AVX-512 tier is verified functionally under Intel SDE; no AVX-512 host was
available to time it.

## Verification

Bit-identity across tiers is a gate, not an aspiration:

```sh
RUNNER=build/fe_run_file FRAMES=500 ./tools/sha256_matrix.sh
```

Every tier within an architecture family must produce the same SHA-256 per
clip. Run it before and after any non-trivial change and diff the output; an
empty diff means no byte-level regression.

`build/fe_test_cross_tier` performs the same check in-process and also reports
cross-tier SNR.

## Weights

`weights/fe.q8` is the production blob (565,108 bytes, 511,754 parameters). To
regenerate it from the upstream ONNX export:

```sh
python3 tools/onnx_to_bin.py --onnx fastenhancer_m_spec.onnx \
        --variant medium --out weights/fe.fp32.bin
python3 tools/quantize_bin.py --in weights/fe.fp32.bin --out weights/fe.q8
```

Weights are per-output-row symmetric int8; activations are per-tensor
asymmetric uint8, recomputed every frame. Both are clamped to `[-127, 127]`
rather than the full int8 range. Discarding one code point makes int16
accumulation provably overflow-free and removes the one value that breaks
cross-tier reproducibility, at a measured cost of 0.000 PESQ to three
decimals.

## Layout

```
include/fe.h          the only public header
src/
  fe_pipeline.c       public ABI
  fe_engine.c         per-frame pipeline
  fe_stft.c fe_fft.c  causal STFT/iSTFT, tiered 1024-point real FFT
  nn/                 conv, GRU, MHSA, activations, GEMM wrappers
  qgemm/{arm,x86}/    per-tier int8 GEMM kernels
  winograd/           F(2,3) microkernels
  fft/                per-tier FFT kernels
tests/                fe_run_file, fe_test_cross_tier, fe_bench_qgemm
tools/                weight pipeline + regression gate
docs/                 architecture, optimization notes
```

## Documentation

- [docs/architecture.md](docs/architecture.md) — layer inventory, shapes, and
  the per-frame dataflow.
- [docs/optimizations.md](docs/optimizations.md) — quantization design, the
  six kernel tiers, fp16 cross-stage storage, op fusion, and measurement
  methodology. It also separates numbers backed by deposited artifacts from
  numbers that exist only as prose in that document.

Both documents are in Korean.

## Acknowledgements

The model, its architecture and its weights are FastEnhancer, by Sunghwan Ahn
et al. — see [aask1357/fastenhancer](https://github.com/aask1357/fastenhancer).
This project is an independent runtime port and is not affiliated with the
authors.

Kernel patterns were informed by ggml/llama.cpp, XNNPACK, oneDNN and ARM
KleidiAI; see `NOTICE` and the Sources table in `docs/optimizations.md`.

Test clips are the RNNoise demo samples, fetched from their original location
by `testaudio/fetch.sh`.

## License

MIT. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
