#!/usr/bin/env python3
"""
quantize_bin.py — Convert an explicit fp32 intermediate blob to a q8
production weights blob.

Replicates fe_qgemm_pack_W exactly:
  * Per-output-channel symmetric int8: scale[n] = max|W[n,:]| / 127 (or 1.0 if 0)
  * Quantize: q[n,k] = clamp(lroundf(W[n,k] / scale[n]), -127, 127)
       lroundf = round-half-away-from-zero
  * Pack into DOTPROD layout: packed[block_n][k4_group][lane=8][k_lane=4]
       always DOTPROD (not I8MM) — the C loader does the I8MM repack on I8MM build.

Layer iteration matches fe_load_weights() in fe_weights.c. FP32 stays for:
  * All biases (always fp32, consumed by dequant)
  * enc_pre.weight (consumed by fe_strided_conv1d via weight_packed_fp32)
  * dec_post_up.weight (consumed by fe_conv_transpose1d)
  * positional encoding (consumed as fp32 in attention residual add)

Output format (little-endian, byte-packed):
    magic   : 8B  = b"FM_W8_03"  (GRU weights are prepacked int8)
    version : u32 = 1
    (header total = 12 bytes)
    <payload — concatenated per layer in fe_load_weights order>

The C loader (`fe_load_weights`) accepts this FM_W8_03 q8 format only.

Requires Python 3.9+; annotations are postponed via `__future__`.
"""
from __future__ import annotations
import argparse
import struct

np = None


def load_dependencies() -> None:
    global np
    if np is not None:
        return
    import numpy as _np
    np = _np

# --- Architecture constants (must match fe_config_medium.h) --------------
FE_C1            = 96
FE_F1            = 128
FE_C2            = 72
FE_F2            = 72
FE_RF_BLOCKS     = 4
FE_ENC_BLOCKS    = 3
FE_DEC_BLOCKS    = FE_ENC_BLOCKS
FE_STRIDE        = 4
FE_STRIDED_CI    = FE_STRIDE * 2    # 8
FE_ENC_K0        = 8
FE_ENC_K         = 3
FE_GRU_DIM       = FE_C2
FE_GRU_GATES     = 3
FE_QGEMM_NR      = 8


# --- Round-half-away-from-zero, matching C lroundf ----------------------
def lround_f(x: np.ndarray) -> np.ndarray:
    """Match C lroundf (halfway away from zero)."""
    return np.where(x >= 0,
                    np.floor(x + 0.5).astype(np.int64),
                    -np.floor(-x + 0.5).astype(np.int64))


def pack_W_dotprod(W: np.ndarray) -> tuple[bytes, np.ndarray]:
    """
    Replicate fe_qgemm_pack_W (DOTPROD layout, pre-I8MM repack):
      W is [N, K] fp32 (PyTorch [out, in] convention).
      Returns: (packed_bytes int8, scales_fp32 [N]).
    """
    assert W.dtype == np.float32 and W.ndim == 2
    N, K = W.shape
    NR = FE_QGEMM_NR
    Np = (N + NR - 1) // NR * NR
    k4g = (K + 3) // 4

    # Per-row scale
    max_abs = np.maximum(np.abs(W).max(axis=1), 0.0)
    scales = np.where(max_abs == 0.0,
                      np.float32(1.0),
                      (max_abs / 127.0).astype(np.float32))

    # Quantize (matches C exactly: `lroundf(val / scale)` — division, one rounding step,
    # NOT `val * (1/scale)` which double-rounds and breaks bit-id on halfway cases).
    # Lower clamp at -127 (not -128) so the vpsignb trick on AVX2 stays
    # saturation-free: psignb cannot negate -128 (i8 wrap to -128 with wrong sign).
    qf = (W.astype(np.float32) / scales[:, None].astype(np.float32)).astype(np.float32)
    qi = np.clip(lround_f(qf), -127, 127).astype(np.int8)  # [N, K]

    # Pad to multiples of NR (rows) and 4 (k_lane)
    Kp = k4g * 4
    qpad = np.zeros((Np, Kp), dtype=np.int8)
    qpad[:N, :K] = qi

    # Pack: packed[block_n][k4_group][lane=NR][k_lane=4]
    # iteration order matches fe_qgemm_pack_W in qgemm_pack.c
    out = np.empty((Np // NR, k4g, NR, 4), dtype=np.int8)
    for bn in range(Np // NR):
        for g in range(k4g):
            for lane in range(NR):
                n_global = bn * NR + lane
                kbase = g * 4
                out[bn, g, lane, :] = qpad[n_global, kbase:kbase + 4]
    return out.tobytes(), scales.astype(np.float32)


# --- Walker matching fe_load_weights iteration --------------------------
class FP32Walker:
    """Reads the existing fp32 .bin one tensor at a time."""
    def __init__(self, buf: bytes):
        self.buf = buf
        self.off = 0

    def take(self, count: int) -> np.ndarray:
        """Take `count` floats as fp32."""
        nbytes = count * 4
        if self.off + nbytes > len(self.buf):
            raise RuntimeError(
                f"unexpected EOF: need {nbytes} more at off {self.off}, "
                f"have {len(self.buf) - self.off}")
        a = np.frombuffer(self.buf, dtype='<f4',
                          count=count, offset=self.off).copy()
        self.off += nbytes
        return a


# --- Convert fp32 .bin → q8 blob ----------------------------------------
def convert(fp32_path: str, out_path: str) -> None:
    load_dependencies()
    with open(fp32_path, 'rb') as f:
        fp32 = f.read()
    w = FP32Walker(fp32)
    out = bytearray()

    # 12-byte header: magic + version=1
    out += b"FM_W8_03"
    out += struct.pack("<I", 1)

    def emit_int8_layer(W_fp32: np.ndarray, N: int, K: int,
                        bias: np.ndarray | None) -> None:
        """[N,K] fp32 → int8 packed + scales [+ bias]."""
        assert W_fp32.shape == (N * K,) or W_fp32.shape == (N, K)
        W = W_fp32.reshape(N, K).astype(np.float32)
        packed, scales = pack_W_dotprod(W)
        out.extend(packed)
        out.extend(scales.tobytes())
        if bias is not None:
            out.extend(bias.astype(np.float32).tobytes())

    def emit_fp32_layer(W_fp32: np.ndarray, bias: np.ndarray) -> None:
        """Verbatim fp32 weight + bias (for enc_pre and dec_post_up)."""
        out.extend(W_fp32.astype(np.float32).tobytes())
        out.extend(bias.astype(np.float32).tobytes())

    # 1) enc_pre — Conv1d(in=FE_STRIDE*2=8, out=FE_C1, kernel=FE_ENC_K0/FE_STRIDE=2)
    enc_pre_in, enc_pre_out, enc_pre_k = FE_STRIDE * 2, FE_C1, FE_ENC_K0 // FE_STRIDE
    enc_pre_w  = w.take(enc_pre_out * enc_pre_in * enc_pre_k)
    enc_pre_b  = w.take(enc_pre_out)
    emit_fp32_layer(enc_pre_w, enc_pre_b)

    # 2) enc[0..2] — Conv1d k=3 (C1→C1)
    for _ in range(FE_ENC_BLOCKS):
        wt = w.take(FE_C1 * FE_C1 * FE_ENC_K)   # [Co, Ci, K]
        bs = w.take(FE_C1)
        # Quantize as [N=Co, K=Ci*K]
        emit_int8_layer(wt, FE_C1, FE_C1 * FE_ENC_K, bs)

    # 3) rf_pre_lin — Linear F1 → F2, no bias
    wt = w.take(FE_F2 * FE_F1)
    emit_int8_layer(wt, FE_F2, FE_F1, None)

    # 4) rf_pre_conv — Conv1d k=1 (C1→C2)
    wt = w.take(FE_C2 * FE_C1 * 1)
    bs = w.take(FE_C2)
    emit_int8_layer(wt, FE_C2, FE_C1, bs)

    # 5) RNNFormer blocks
    D = FE_GRU_DIM
    for blk in range(FE_RF_BLOCKS):
        Wih = w.take(3 * D * D);   bih = w.take(3 * D)
        Whh = w.take(3 * D * D);   bhh = w.take(3 * D)
        # Runtime uses W8A8 GRU weights, so the blob stores the same packed form
        # directly and keeps only fp32 biases.
        emit_int8_layer(Wih, 3 * D, D, bih)
        emit_int8_layer(Whh, 3 * D, D, bhh)
        # rnn_fc
        wt = w.take(FE_C2 * FE_C2); bs = w.take(FE_C2)
        emit_int8_layer(wt, FE_C2, FE_C2, bs)
        # attn.qkv (no bias)
        wt = w.take(3 * FE_C2 * FE_C2)
        emit_int8_layer(wt, 3 * FE_C2, FE_C2, None)
        # attn_fc
        wt = w.take(FE_C2 * FE_C2); bs = w.take(FE_C2)
        emit_int8_layer(wt, FE_C2, FE_C2, bs)
        # pe on block 0
        if blk == 0:
            pe = w.take(FE_F2 * FE_C2)
            out.extend(pe.astype(np.float32).tobytes())

    # 6) rf_post_lin — Linear F2 → F1, no bias
    wt = w.take(FE_F1 * FE_F2)
    emit_int8_layer(wt, FE_F1, FE_F2, None)

    # 7) rf_post_conv — Conv1d k=1 (C2→C1)
    wt = w.take(FE_C1 * FE_C2 * 1); bs = w.take(FE_C1)
    emit_int8_layer(wt, FE_C1, FE_C2, bs)

    # 8) Decoder blocks
    for _ in range(FE_DEC_BLOCKS):
        # dec_1x1 (Conv1d k=1, 2*C1→C1)
        wt = w.take(FE_C1 * 2 * FE_C1 * 1); bs = w.take(FE_C1)
        emit_int8_layer(wt, FE_C1, 2 * FE_C1, bs)
        # dec_3x3 (Conv1d k=3, C1→C1)
        wt = w.take(FE_C1 * FE_C1 * FE_ENC_K); bs = w.take(FE_C1)
        emit_int8_layer(wt, FE_C1, FE_C1 * FE_ENC_K, bs)

    # 9) dec_post_1x1 — Conv1d k=1 (2*C1→C1)
    wt = w.take(FE_C1 * 2 * FE_C1 * 1); bs = w.take(FE_C1)
    emit_int8_layer(wt, FE_C1, 2 * FE_C1, bs)

    # 10) dec_post_up — ConvT1d (in=C1, out=2, kernel=ENC_K0=8, stride=STRIDE=4) → fp32
    convt_w = w.take(FE_C1 * 2 * FE_ENC_K0)   # [Ci, Co, K]
    convt_b = w.take(2)
    emit_fp32_layer(convt_w, convt_b)

    if w.off != len(fp32):
        raise RuntimeError(
            f"didn't consume entire fp32 file: off={w.off}, size={len(fp32)}")

    with open(out_path, 'wb') as f:
        f.write(bytes(out))

    print(f"fp32 in  : {len(fp32):,} bytes  ({fp32_path})")
    print(f"int8 out : {len(out):,} bytes  ({out_path})")
    print("format   : FM_W8_03")
    print(f"ratio    : {len(out) / len(fp32):.3f}× "
          f"({len(fp32) / len(out):.2f}× smaller)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  dest="inp",  required=True, help="fp32 .bin path")
    ap.add_argument("--out", dest="outp", required=True,
                    help="q8 blob path (for example weights/fe.q8)")
    args = ap.parse_args()
    convert(args.inp, args.outp)


if __name__ == "__main__":
    main()
