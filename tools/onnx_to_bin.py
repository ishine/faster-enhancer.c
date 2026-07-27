#!/usr/bin/env python3
"""
onnx_to_bin.py — FastEnhancer 48 kHz spec2spec ONNX → flat fp32
                 intermediate blob for the faster-enhancer.c q8 export pipeline.

The output order is consumed by tools/quantize_bin.py, which emits the q8
blob parsed by fe_load_weights(). The order MUST match exactly.

Usage:
    python tools/onnx_to_bin.py \
        --variant medium \
        --onnx path/to/fastenhancer_m_spec.onnx \
        --out  weights/fe.fp32.bin
"""
import argparse
import os
import sys

np = None
onnx = None
numpy_helper = None


def load_dependencies():
    global np, onnx, numpy_helper
    if np is not None:
        return
    import numpy as _np
    import onnx as _onnx
    from onnx import numpy_helper as _numpy_helper
    np = _np
    onnx = _onnx
    numpy_helper = _numpy_helper


# ---------------------------------------------------------------------------
# Architecture constants — this repo's runtime/q8 exporter is Medium-only.
# These values must match fe_config_medium.h and tools/quantize_bin.py.
# ---------------------------------------------------------------------------
VARIANT_PARAMS = {
    #              C1  C2  F2  RF  ENC
    "medium": dict(C1=96,  C2=72, F2=72, RF_BLOCKS=4, ENC_BLOCKS=3),
}

# Fixed Medium constants
FE_F1          = 128      # 512/4
FE_NUM_HEADS   = 4
FE_DEC_BLOCKS  = None     # mirrors ENC_BLOCKS
FE_ENC_K0      = 8
FE_ENC_K       = 3
FE_STRIDE      = 4

# Filled in by export() once --variant is parsed
FE_C1 = FE_C2 = FE_F2 = FE_HEAD_DIM = FE_RF_BLOCKS = FE_ENC_BLOCKS = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_initializers(path):
    """Load ONNX model and return dict: name -> np.ndarray."""
    model = onnx.load(path)
    out = {}
    for init in model.graph.initializer:
        out[init.name] = numpy_helper.to_array(init)
    return out, model


def find_matmul_weight(inits, model, node_name_substr):
    """Find the second input (weight) of the MatMul node whose name matches."""
    for n in model.graph.node:
        if n.op_type == "MatMul" and node_name_substr in n.name:
            wname = n.input[1]
            if wname in inits:
                return inits[wname], wname
    raise KeyError(f"MatMul node not found: {node_name_substr}")


def find_gru_inputs(inits, model, block_idx):
    """Locate the (W_ih, W_hh, bias) initializers for the GRU at block_idx."""
    target = f"/rf_block.{block_idx}/rnn/GRU"
    for n in model.graph.node:
        if n.op_type == "GRU" and n.name == target:
            return (inits[n.input[1]], inits[n.input[2]], inits[n.input[3]])
    raise KeyError(f"GRU node not found for block {block_idx}")


def write(f, arr, label, expected_shape=None):
    """Cast to float32 contiguous, write, log shape and count."""
    a = np.ascontiguousarray(arr, dtype=np.float32)
    if expected_shape is not None:
        if list(a.shape) != list(expected_shape):
            raise ValueError(
                f"Shape mismatch for {label}: got {list(a.shape)}, "
                f"expected {list(expected_shape)}"
            )
    f.write(a.tobytes())
    n = a.size
    print(f"  {label:45s} shape={str(list(a.shape)):20s} numel={n:7d}")
    return n


# ---------------------------------------------------------------------------
# Main export
# ---------------------------------------------------------------------------
def export(args):
    load_dependencies()
    # Resolve variant
    global FE_C1, FE_C2, FE_F2, FE_HEAD_DIM, FE_RF_BLOCKS, FE_ENC_BLOCKS, FE_DEC_BLOCKS
    p = VARIANT_PARAMS[args.variant]
    FE_C1, FE_C2, FE_F2 = p["C1"], p["C2"], p["F2"]
    FE_RF_BLOCKS  = p["RF_BLOCKS"]
    FE_ENC_BLOCKS = p["ENC_BLOCKS"]
    FE_DEC_BLOCKS = FE_ENC_BLOCKS
    FE_HEAD_DIM   = FE_C2 // FE_NUM_HEADS
    print(f"Variant: {args.variant}  (C1={FE_C1}, C2={FE_C2}, F2={FE_F2}, "
          f"ENC_BLOCKS={FE_ENC_BLOCKS}, RF_BLOCKS={FE_RF_BLOCKS}, "
          f"HEAD_DIM={FE_HEAD_DIM})\n")

    inits, model = load_initializers(args.onnx)
    print(f"Loaded {len(inits)} initializers from {args.onnx}\n")

    total = 0
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    with open(args.out, "wb") as f:

        # ===== 1. Encoder PreNet (StridedConv reshaped: in=8, out=C1, k=2) =====
        print("=== Encoder PreNet ===")
        total += write(f, inits["enc_pre.0.weight"],
                       "enc_pre.weight", [FE_C1, FE_STRIDE * 2, FE_ENC_K0 // FE_STRIDE])
        total += write(f, inits["enc_pre.0.bias"],
                       "enc_pre.bias", [FE_C1])

        # ===== 2. Encoder Blocks (Conv1d C1->C1, k=3) =====
        print("\n=== Encoder Blocks ===")
        for i in range(FE_ENC_BLOCKS):
            total += write(f, inits[f"encoder.{i}.0.weight"],
                           f"encoder[{i}].weight", [FE_C1, FE_C1, FE_ENC_K])
            total += write(f, inits[f"encoder.{i}.0.bias"],
                           f"encoder[{i}].bias", [FE_C1])

        # ===== 3. RNNFormer PreNet =====
        print("\n=== RNNFormer PreNet ===")
        # rf_pre.0: Linear(F1=128 -> F2)
        # ONNX MatMul stores transposed: [F1, F2] -> PyTorch needs [F2, F1]
        w, wname = find_matmul_weight(inits, model, "/rf_pre/rf_pre.0/MatMul")
        # ONNX shape is [F1, F2]; PyTorch nn.Linear wants [F2, F1].
        rf_pre_lin_w = w.T.copy()
        total += write(f, rf_pre_lin_w,
                       f"rf_pre_lin.weight (from {wname})", [FE_F2, FE_F1])
        # rf_pre.1: Conv1d (C1 -> C2, k=1)
        total += write(f, inits["rf_pre.1.weight"],
                       "rf_pre_conv.weight", [FE_C2, FE_C1, 1])
        total += write(f, inits["rf_pre.1.bias"],
                       "rf_pre_conv.bias", [FE_C2])

        # ===== 4. RNNFormer Blocks =====
        print("\n=== RNNFormer Blocks ===")
        for i in range(FE_RF_BLOCKS):
            print(f"\n  --- Block {i} ---")
            # GRU: ONNX layout is [1, 3*hidden, input/hidden]
            # PyTorch convention: [3*hidden, input] / [3*hidden, hidden]
            #
            W_ih, W_hh, bias = find_gru_inputs(inits, model, i)
            #
            # ONNX GRU gate order is [z, r, h] (update, reset, hidden).
            # PyTorch GRU gate order is [r, z, n] (reset, update, new).
            # The C engine assumes PyTorch ordering, so swap the first
            # two D-sized blocks. (z=h=n have matching semantics; only
            # the first two slots differ.)
            #
            def reorder_zrh_to_rzn(arr, D):
                # arr shape [3*D, ...]: [z, r, h] → [r, z, h]
                z, r, h = arr[:D], arr[D:2*D], arr[2*D:]
                return np.concatenate([r, z, h], axis=0)

            W_ih_sq = reorder_zrh_to_rzn(W_ih.reshape(3 * FE_C2, FE_C2), FE_C2)
            W_hh_sq = reorder_zrh_to_rzn(W_hh.reshape(3 * FE_C2, FE_C2), FE_C2)
            b_flat  = bias.reshape(-1)                       # [120]
            b_ih    = reorder_zrh_to_rzn(b_flat[:3 * FE_C2], FE_C2)
            b_hh    = reorder_zrh_to_rzn(b_flat[3 * FE_C2:], FE_C2)
            total += write(f, W_ih_sq,
                           f"rf[{i}].gru.W_ih", [3 * FE_C2, FE_C2])
            total += write(f, b_ih,
                           f"rf[{i}].gru.b_ih", [3 * FE_C2])
            total += write(f, W_hh_sq,
                           f"rf[{i}].gru.W_hh", [3 * FE_C2, FE_C2])
            total += write(f, b_hh,
                           f"rf[{i}].gru.b_hh", [3 * FE_C2])

            # rnn_fc: Gemm with transB=1 -> weight already in PyTorch convention [out, in]
            rnn_fc_w = inits[f"/rf_block.{i}/rnn_fc/MatMul_weight"]
            total += write(f, rnn_fc_w,
                           f"rf[{i}].rnn_fc.weight", [FE_C2, FE_C2])
            total += write(f, inits[f"rf_block.{i}.rnn_fc.bias"],
                           f"rf[{i}].rnn_fc.bias", [FE_C2])

            # attn.qkv: MatMul -> stored [C2, 3*C2] -> PyTorch [3*C2, C2]
            qkv_w, qkv_name = find_matmul_weight(
                inits, model, f"/rf_block.{i}/attn/qkv/MatMul")
            qkv_pt = qkv_w.T.copy()
            total += write(f, qkv_pt,
                           f"rf[{i}].attn.qkv.weight (from {qkv_name})",
                           [3 * FE_C2, FE_C2])

            # attn_fc: Gemm with transB=1 -> already PyTorch form
            attn_fc_w = inits[f"/rf_block.{i}/attn_fc/MatMul_weight"]
            total += write(f, attn_fc_w,
                           f"rf[{i}].attn_fc.weight", [FE_C2, FE_C2])
            total += write(f, inits[f"rf_block.{i}.attn_fc.bias"],
                           f"rf[{i}].attn_fc.bias", [FE_C2])

            # Positional embedding (block 0 only)
            if i == 0:
                total += write(f, inits["rf_block.0.pe"],
                               f"rf[{i}].pe", [FE_F2, FE_C2])

        # ===== 5. RNNFormer PostNet =====
        print("\n=== RNNFormer PostNet ===")
        # rf_post.0: Linear(F2 -> F1=128). ONNX MatMul: [F2, F1] -> PyTorch [F1, F2]
        w, wname = find_matmul_weight(inits, model, "/rf_post/rf_post.0/MatMul")
        rf_post_lin_w = w.T.copy()
        total += write(f, rf_post_lin_w,
                       f"rf_post_lin.weight (from {wname})", [FE_F1, FE_F2])
        total += write(f, inits["rf_post.1.weight"],
                       "rf_post_conv.weight", [FE_C1, FE_C2, 1])
        total += write(f, inits["rf_post.1.bias"],
                       "rf_post_conv.bias", [FE_C1])

        # ===== 6. Decoder Blocks (each: 1x1 then 3x3) =====
        print("\n=== Decoder Blocks ===")
        for i in range(FE_DEC_BLOCKS):
            total += write(f, inits[f"decoder.{i}.0.weight"],
                           f"dec[{i}].1x1.weight", [FE_C1, 2 * FE_C1, 1])
            total += write(f, inits[f"decoder.{i}.0.bias"],
                           f"dec[{i}].1x1.bias", [FE_C1])
            total += write(f, inits[f"decoder.{i}.2.weight"],
                           f"dec[{i}].3x3.weight", [FE_C1, FE_C1, FE_ENC_K])
            total += write(f, inits[f"decoder.{i}.2.bias"],
                           f"dec[{i}].3x3.bias", [FE_C1])

        # ===== 7. Decoder PostNet =====
        print("\n=== Decoder PostNet ===")
        total += write(f, inits["dec_post.0.weight"],
                       "dec_post_1x1.weight", [FE_C1, 2 * FE_C1, 1])
        total += write(f, inits["dec_post.0.bias"],
                       "dec_post_1x1.bias", [FE_C1])
        # ConvTranspose1d: weight shape [in_ch=C1, out_ch=2, k=8]
        total += write(f, inits["dec_post.2.weight"],
                       "dec_post_up.weight", [FE_C1, 2, FE_ENC_K0])
        total += write(f, inits["dec_post.2.bias"],
                       "dec_post_up.bias", [2])

    size = os.path.getsize(args.out)
    print(f"\n{'=' * 60}")
    print(f"Total floats written: {total:,}")
    print(f"File size: {size:,} bytes ({size / 1024:.1f} KiB)")
    print(f"Expected:  {total * 4:,} bytes")
    print(f"Output: {args.out}")
    if size != total * 4:
        print("WARNING: file size mismatch!")
        return 1
    return 0


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True,
                    help="Path to *_spec.onnx (spec2spec variant)")
    ap.add_argument("--variant", default="medium",
                    choices=["medium"],
                    help="Model variant — only medium is supported by this runtime/q8 exporter")
    ap.add_argument("--out",  default=None,
                    help="Output binary path (default: weights/fe.fp32.bin)")
    args = ap.parse_args()
    if args.out is None:
        args.out = "weights/fe.fp32.bin"
    sys.exit(export(args))
