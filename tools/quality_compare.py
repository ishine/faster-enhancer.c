#!/usr/bin/env python3
"""
quality_compare.py — PESQ-WB + STOI + eSTOI + SNR between two WAV files.

PESQ requires 16 kHz, so we resample with scipy.signal.resample_poly.
The two inputs must have the same sample rate (typically 48 kHz from our
engine output).

The engine's STFT introduces a fixed analysis latency (N_FFT - hop = 704
samples = 14.67 ms @ 48 kHz) relative to the ONNX reference. STOI / eSTOI
/ SNR are sample-alignment-sensitive, so a 704-sample skew collapses them
(corr ~0.05, SNR negative) even when the two signals are perceptually
near-identical. We therefore cross-correlate to estimate the integer lag
and align before scoring. PESQ does its own internal alignment but we feed
it the aligned pair too for consistency. Pass --no-align to disable.

usage: quality_compare.py <ref.wav> <test.wav> [--no-align] [--max-lag-ms N]

Requires: numpy, scipy, pystoi, pesq.
"""
import sys

argv = [a for a in sys.argv[1:]]
do_align = "--no-align" not in argv
argv = [a for a in argv if a != "--no-align"]
max_lag_ms = 100.0
if "--max-lag-ms" in argv:
    i = argv.index("--max-lag-ms")
    max_lag_ms = float(argv[i + 1])
    del argv[i:i + 2]

if len(argv) < 2:
    print("usage: quality_compare.py <ref.wav> <test.wav> "
          "[--no-align] [--max-lag-ms N]")
    sys.exit(1)

ref_path = argv[0]
test_path = argv[1]

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly, correlate
from pystoi import stoi
from pesq import pesq

sr_ref, ref = wavfile.read(ref_path)
sr_t, test = wavfile.read(test_path)
if sr_ref != sr_t:
    print(f"sample rate mismatch: {sr_ref} vs {sr_t}")
    sys.exit(1)
sr = sr_ref

# Cast to float32, normalize int16 to [-1, 1]
if ref.dtype == np.int16:
    ref = ref.astype(np.float32) / 32768.0
if test.dtype == np.int16:
    test = test.astype(np.float32) / 32768.0
ref = ref.astype(np.float32)
test = test.astype(np.float32)

# Align length
n = min(len(ref), len(test))
ref = ref[:n]
test = test[:n]


def estimate_lag(ref, test, sr, max_lag_ms):
    """Integer sample lag that best aligns `test` onto `ref`, via FFT
    cross-correlation over a mid segment. Positive lag => test is delayed
    relative to ref (drop test's leading `lag` samples to align)."""
    n = min(len(ref), len(test))
    seg = min(int(2.0 * sr), n)          # 2 s window is plenty
    start = (n - seg) // 2
    a = test[start:start + seg].astype(np.float64)
    b = ref[start:start + seg].astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    if np.allclose(a, 0) or np.allclose(b, 0):
        return 0
    xc = correlate(a, b, mode="full", method="fft")
    lags = np.arange(-(len(b) - 1), len(a))
    max_lag = int(max_lag_ms * 1e-3 * sr)
    mask = np.abs(lags) <= max_lag
    return int(lags[mask][np.argmax(xc[mask])])


lag = 0
if do_align:
    lag = estimate_lag(ref, test, sr, max_lag_ms)
    if lag > 0:        # test delayed: drop its leading `lag` samples
        test = test[lag:]
        ref = ref[:len(test)]
    elif lag < 0:      # ref delayed: drop its leading `-lag` samples
        ref = ref[-lag:]
        test = test[:len(ref)]
    m = min(len(ref), len(test))
    ref = ref[:m]
    test = test[:m]
    n = m

# STOI / eSTOI at native sample rate
stoi_v = stoi(ref, test, sr, extended=False)
estoi_v = stoi(ref, test, sr, extended=True)

# SNR
diff = test - ref
noise_p = float(np.mean(diff ** 2))
sig_p = float(np.mean(ref ** 2))
snr_db = 10.0 * np.log10(sig_p / (noise_p + 1e-12)) if noise_p > 0 else float('inf')
max_d = float(np.abs(diff).max())

# PESQ-WB needs 16 kHz; resample if needed
if sr != 16000:
    g = np.gcd(sr, 16000)
    ref_16k = resample_poly(ref, 16000 // g, sr // g).astype(np.float32)
    test_16k = resample_poly(test, 16000 // g, sr // g).astype(np.float32)
else:
    ref_16k, test_16k = ref, test

try:
    pesq_wb = pesq(16000, ref_16k, test_16k, 'wb')
    pesq_str = f"{pesq_wb:.3f}"
except Exception as e:
    pesq_str = f"FAIL ({e})"

print(f"  ref:  {ref_path}")
print(f"  test: {test_path}")
print(f"  sr={sr} samples={n}")
if do_align:
    print(f"  align:   lag={lag} samp ({lag / sr * 1e3:+.2f} ms)")
else:
    print(f"  align:   DISABLED (--no-align)")
print(f"  STOI:    {stoi_v:.6f}")
print(f"  eSTOI:   {estoi_v:.6f}")
print(f"  SNR:     {snr_db:.2f} dB")
print(f"  PESQ-WB: {pesq_str}")
print(f"  max|d|:  {max_d:.6f}")
