#!/usr/bin/env python3
"""
onnx_reference_causal.py — fp32 ONNX reference on the ENGINE's analysis grid.

Why this exists (vs a default center=True reference)
--------------------------------------------
`onnx_reference.py` calls `torch.stft` / `torch.istft` without `center=`,
so PyTorch's default `center=True` applies: the signal is reflect-padded by
n_fft//2 = 512 and frame t is *centered* on sample t*hop.

The C engine is a causal streaming STFT. Frame t is
`[cache_stft(704) | in_t(320)]`, i.e. it spans input samples
`[t*hop - 704, t*hop + 320)`. That is `center=False` with a 704-sample
zero left-history.

704 - 512 = 192 samples of analysis-grid offset, and 192 is not a multiple
of hop (320), so it does not cancel. Running the *same fp32 model* on two
grids 192 samples apart already costs ~23 dB SNR / ~4.29 PESQ-WB — the same
order as the number the repo used to report as "q8 vs fp32 fidelity".
That comparison therefore measured framing phase, not quantization.

This script replicates the engine's framing exactly, so the only remaining
difference against the engine output is arithmetic (int8 kernels, SIMD
polynomial transcendentals, fp16 state) — which is what we want to report.

Contract mirrored from src/fe_stft.c:
  - periodic Hann, w[n] = 0.5 - 0.5*cos(2*pi*n/N)
  - frame t = padded[t*H : t*H + N], padded = [704 zeros | audio]
  - synthesis window = w / win_sq_sum, win_sq_sum from the 7-frame fold
    sliced at (K-1)*H with K = ceil(N/H) = 4
  - overlap-add with hop H; output sample n aligns 1:1 with the engine's
    output stream (both carry the same 704-sample algorithmic delay)

Power compression / decompression is NOT done here: the spec2spec ONNX
graph does it internally (Clip/Pow nodes), exactly as the engine does it
in fe_stft/fe_istft.

usage: onnx_reference_causal.py <input.f32> <model.onnx> <out.wav> [num_frames]
"""
import sys

import numpy as np
import onnxruntime as ort
from scipy.io import wavfile

SR = 48000
N_FFT = 1024
HOP = 320
CACHE = N_FFT - HOP          # 704, FE_CACHE_LEN


def synthesis_window():
    """Mirror fe_stft_init(): w / win_sq_sum over the overlap region."""
    n = np.arange(N_FFT)
    w = 0.5 - 0.5 * np.cos(2.0 * np.pi * n / N_FFT)      # periodic Hann
    K = (N_FFT + HOP - 1) // HOP                          # 4
    num_frames = 2 * K - 1                                # 7
    fold = np.zeros(num_frames * HOP + N_FFT)
    for f in range(num_frames):
        fold[f * HOP: f * HOP + N_FFT] += w * w
    win_sq_sum = fold[(K - 1) * HOP: (K - 1) * HOP + N_FFT]
    # NOTE: the engine also folds 1/N here because fe_irfft omits it.
    # np.fft.irfft already applies 1/N, so we must NOT fold it again.
    return w, w / win_sq_sum


def main(in_f32, onnx_path, out_wav, num_frames=None):
    audio = np.fromfile(in_f32, dtype=np.float32)
    audio = np.clip(audio, -1.0, 1.0)
    T = len(audio) // HOP
    if num_frames is not None:
        T = min(T, int(num_frames))
    print(f"input: {len(audio)} samples ({len(audio)/SR:.2f}s) -> {T} frames")

    # Causal left-history: the engine starts with cache_stft = 0.
    padded = np.concatenate([np.zeros(CACHE, dtype=np.float64),
                             audio.astype(np.float64)])
    need = (T - 1) * HOP + N_FFT
    if len(padded) < need:
        padded = np.concatenate([padded, np.zeros(need - len(padded))])

    w_ana, w_syn = synthesis_window()

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess = ort.InferenceSession(onnx_path, sess_options=so,
                                providers=['CPUExecutionProvider'])
    cache_names = [x.name for x in sess.get_inputs()
                   if x.name.startswith('cache_in_')]
    caches = {}
    for n in cache_names:
        shape = [d if isinstance(d, int) else 1
                 for d in next(x.shape for x in sess.get_inputs() if x.name == n)]
        caches[n] = np.zeros(shape, dtype=np.float32)

    ola = np.zeros(T * HOP + N_FFT)
    for t in range(T):
        frame = padded[t * HOP: t * HOP + N_FFT] * w_ana
        spec = np.fft.rfft(frame)                          # 513 bins
        spec_in = np.stack([spec.real, spec.imag], axis=-1)
        spec_in = spec_in.reshape(1, N_FFT // 2 + 1, 1, 2).astype(np.float32)

        feed = {'spec_in': spec_in}
        feed.update(caches)
        outs = sess.run(None, feed)
        for j, n in enumerate(cache_names):
            caches[n] = outs[j + 1]

        so_ = outs[0].reshape(N_FFT // 2 + 1, 2).astype(np.float64)
        rec = np.fft.irfft(so_[:, 0] + 1j * so_[:, 1], n=N_FFT)
        ola[t * HOP: t * HOP + N_FFT] += rec * w_syn

        if (t + 1) % 1000 == 0:
            print(f"  {t+1}/{T} frames", flush=True)

    out = np.clip(ola[: T * HOP], -1.0, 1.0)
    wavfile.write(out_wav, SR, np.clip(out * 32767.0, -32768, 32767).astype(np.int16))
    print(f"wrote: {out_wav}  ({T * HOP} samples)")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3],
         sys.argv[4] if len(sys.argv) > 4 else None)
