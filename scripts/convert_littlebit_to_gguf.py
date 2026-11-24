#!/usr/bin/env python3
"""Convert a LittleBit / LittleBit-2 checkpoint to a GGUF file whose factorised layers are Q1_0 tensors.

Two steps:
  1. The uncompressed tensors (embeddings, output head, norms) plus tokenizer and hyperparameters are
     converted by llama.cpp's stock convert_hf_to_gguf.py from a temporary "skeleton" checkpoint.
  2. Each factorised projection is added as three tensors per path (primary, and "_r" for residual):
        <base>.lb_g   F32   [in]         input scale v2
        <base>.lb_v   Q1_0  [in, r_pad]  sign(V), block scale d = v1*u2 of that row (0 for padded rows)
        <base>.lb_u   Q1_0  [r_pad, out] sign(U), block scale d = u1 of that row
     so that  y = U' (V' (g * x))  with ordinary Q1_0 matrix multiplies.
     Folding the scales into the Q1_0 block scale is the idea described in
     github.com/GeniusDevam/littlebit-gguf docs/DESIGN.md.

Usage:
  convert_littlebit_to_gguf.py CHECKPOINT_DIR OUT.gguf --llama-cpp /path/to/llama.cpp
      [--embd-type f16|q8_0] [--embd-quant Q4_K_M --quantize-bin /path/to/llama-quantize] [--unpadded-v]

With --unpadded-v, lb_v is stored at the true rank r instead of r_pad (littlebit.format_version = 2).
lb_u stays padded because the rank is its Q1_0 block dimension; the loader zero-pads the latent.
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

BLOCK = 128
PROJ = {
    "self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k", "self_attn.v_proj": "attn_v",
    "self_attn.o_proj": "attn_output", "mlp.gate_proj": "ffn_gate", "mlp.up_proj": "ffn_up",
    "mlp.down_proj": "ffn_down",
}


def to_q1_0(packed: np.ndarray, cols: int, row_scale: np.ndarray, rows_out: int, cols_out: int) -> np.ndarray:
    """LittleBit packed sign rows -> raw Q1_0 bytes [rows_out, cols_out/128*18].

    packed: int32 [rows, ceil(cols/32)], lsb first, bit 1 = -1. Q1_0 wants bit 1 = +d, so bits are inverted.
    Rows and columns beyond the source are padding: padded rows get d = 0 and padding sign bits are 0.
    """
    rows = packed.shape[0]
    assert cols_out % BLOCK == 0 and rows_out >= rows and cols_out >= cols
    bits = np.zeros((rows_out, cols_out // 8), dtype=np.uint8)
    src = np.ascontiguousarray(packed.astype("<i4")).view(np.uint8)          # [rows, words*4]
    bits[:rows, : src.shape[1]] = ~src
    # columns beyond `cols` are padding: clear their sign bits (the shared format asks writers to)
    if cols % 8:
        bits[:, cols // 8] &= (1 << (cols % 8)) - 1
    bits[:, -(-cols // 8):] = 0
    d = np.zeros(rows_out, dtype=np.float32)
    d[:rows] = row_scale
    d16 = d.astype("<f2")
    out = np.empty((rows_out, cols_out // BLOCK, 18), dtype=np.uint8)
    out[:, :, :2] = d16.view(np.uint8).reshape(rows_out, 1, 2)
    out[:, :, 2:] = bits.reshape(rows_out, cols_out // BLOCK, 16)
    return out.reshape(rows_out, -1)


def q1_0_to_dense(raw: np.ndarray) -> np.ndarray:
    rows = raw.shape[0]
    blocks = raw.reshape(rows, -1, 18)
    d = blocks[:, :, :2].copy().view("<f2").astype(np.float32)
    bits = np.unpackbits(blocks[:, :, 2:], axis=-1, bitorder="little")
    return np.where(bits == 1, d, -d).reshape(rows, -1)


def unpack(packed: np.ndarray, cols: int) -> np.ndarray:
    bits = np.unpackbits(np.ascontiguousarray(packed.astype("<i4")).view(np.uint8), axis=1, bitorder="little")
    return np.where(bits[:, :cols] == 1, -1.0, 1.0).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--llama-cpp", type=Path, required=True)
    ap.add_argument("--embd-type", default="f16", choices=["f32", "f16", "bf16", "q8_0"])
    ap.add_argument("--embd-quant", default=None, metavar="TYPE",
                    help="requantize the uncompressed tensors with llama-quantize, e.g. Q4_K_M or Q6_K")
    ap.add_argument("--quantize-bin", type=Path, default=None, help="path to llama-quantize (for --embd-quant)")
    ap.add_argument("--validate", type=int, default=3, help="number of projections to check numerically")
    ap.add_argument("--unpadded-v", action="store_true",
                    help="store lb_v at the true rank (format version 2); lb_u stays padded to a multiple of 128")
    args = ap.parse_args()

    sys.path.insert(0, str(args.llama_cpp / "gguf-py"))
    import gguf  # noqa: PLC0415

    st_path = args.checkpoint / "model.safetensors"
    lb_cfg = json.loads((args.checkpoint / "littlebit_config.json").read_text())

    with safe_open(st_path, "pt") as st, tempfile.TemporaryDirectory(dir=args.out.parent) as tmp:
        tmp = Path(tmp)
        names = list(st.keys())
        lb_suffixes = ("_packed", "_shape", ".u1", ".u2", ".v1", ".v2", ".u1_R", ".u2_R", ".v1_R", ".v2_R",
                       "._eff_bit_actual", "._eff_bit_target", "._split_dim_final")
        dense = {n: st.get_tensor(n) for n in names if not n.endswith(lb_suffixes)}
        print(f"step 1: {len(dense)} uncompressed tensors -> stock converter ({args.embd_type})")
        save_file(dense, tmp / "model.safetensors", metadata={"format": "pt"})
        del dense
        for f in args.checkpoint.iterdir():
            if f.name != "model.safetensors" and f.is_file():
                (tmp / f.name).symlink_to(f.resolve())
        skeleton = tmp / "skeleton.gguf"
        subprocess.run([sys.executable, str(args.llama_cpp / "convert_hf_to_gguf.py"), str(tmp),
                        "--outfile", str(skeleton), "--outtype", args.embd_type], check=True,
                       stdout=subprocess.DEVNULL)
        if args.embd_quant:
            qbin = args.quantize_bin or args.llama_cpp / "build" / "bin" / "llama-quantize"
            quantized = tmp / "skeleton-quant.gguf"
            subprocess.run([str(qbin), str(skeleton), str(quantized), args.embd_quant], check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            skeleton = quantized

        reader = gguf.GGUFReader(skeleton)
        arch = reader.fields[gguf.Keys.General.ARCHITECTURE].contents()
        writer = gguf.GGUFWriter(args.out, arch=arch)
        for field in reader.fields.values():
            if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
                continue
            vt = field.types[0]
            sub = field.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
            writer.add_key_value(field.name, field.contents(), vt, sub_type=sub)
        writer.add_uint32("littlebit.format_version", 2 if args.unpadded_v else 1)
        writer.add_bool("littlebit.residual", bool(lb_cfg.get("residual", False)))
        writer.add_float32("littlebit.eff_bits", float(lb_cfg.get("eff_bit", 0.0)))
        writer.add_bool("littlebit.use_itq", bool(lb_cfg.get("use_itq", False)))

        for t in reader.tensors:
            writer.add_tensor(t.name, t.data, raw_dtype=t.tensor_type)

        print("step 2: factorised projections -> Q1_0")
        bases = sorted({n[: -len(".U_packed")] for n in names if n.endswith(".U_packed")})
        sign_bytes = scale_range = 0
        max_abs, min_abs = 0.0, float("inf")
        checked = 0
        rng = np.random.default_rng(0)
        for base in bases:
            _, _, layer, proj = base.split(".", 3)                      # model.layers.N.<proj>
            gbase = f"blk.{layer}.{PROJ[proj]}"
            for src_sfx, dst_sfx in (("", ""), ("_R", "_r")):
                if f"{base}.U{src_sfx}_packed" not in names:
                    continue
                f32 = lambda n: st.get_tensor(n).float().numpy().reshape(-1)  # noqa: E731
                u_p = st.get_tensor(f"{base}.U{src_sfx}_packed").numpy()
                v_p = st.get_tensor(f"{base}.V{src_sfx}_packed").numpy()
                out_f, rank = (int(x) for x in st.get_tensor(f"{base}.U{src_sfx}_shape"))
                rank_v, in_f = (int(x) for x in st.get_tensor(f"{base}.V{src_sfx}_shape"))
                assert rank == rank_v and in_f % BLOCK == 0, (base, rank, rank_v, in_f)
                u1, v2 = f32(f"{base}.u1{src_sfx}"), f32(f"{base}.v2{src_sfx}")
                mid = f32(f"{base}.v1{src_sfx}") * f32(f"{base}.u2{src_sfx}")
                for s in (u1, mid):
                    a = np.abs(s[s != 0])
                    max_abs, min_abs = max(max_abs, float(a.max())), min(min_abs, float(a.min()))
                    scale_range += int((a > 65504).sum() + (a < 6.2e-5).sum())   # outside normal fp16 range
                r_pad = -(-rank // BLOCK) * BLOCK
                v_q = to_q1_0(v_p, in_f, mid, rank if args.unpadded_v else r_pad, in_f)   # [r or r_pad, in]
                u_q = to_q1_0(u_p, rank, u1, out_f, r_pad)             # [out, r_pad]
                writer.add_tensor(f"{gbase}.lb_g{dst_sfx}", v2.astype(np.float32))
                writer.add_tensor(f"{gbase}.lb_v{dst_sfx}", v_q, raw_dtype=gguf.GGMLQuantizationType.Q1_0)
                writer.add_tensor(f"{gbase}.lb_u{dst_sfx}", u_q, raw_dtype=gguf.GGMLQuantizationType.Q1_0)
                sign_bytes += v_q.nbytes + u_q.nbytes

                if checked < args.validate and layer in ("0", "17", "35") and proj == "self_attn.k_proj":
                    x = rng.normal(size=(2, in_f)).astype(np.float32)
                    ref = ((((x * v2) @ unpack(v_p, in_f).T) * mid) @ unpack(u_p, rank).T) * u1
                    lat = (x * v2) @ q1_0_to_dense(v_q).T
                    lat = np.pad(lat, ((0, 0), (0, r_pad - lat.shape[1])))    # no-op for the padded layout
                    got = lat @ q1_0_to_dense(u_q).T
                    err = float(np.abs(got - ref).max() / np.abs(ref).max())
                    print(f"  check {gbase}{dst_sfx}: rank {rank}->{r_pad}, max rel diff vs reference {err:.2e}")
                    assert err < 2e-3, "converted tensors do not reproduce the reference forward"
                    checked += 1

        print(f"  {len(bases)} projections, sign tensors {sign_bytes / 1e6:.0f} MB, "
              f"|scale| range [{min_abs:.3g}, {max_abs:.3g}], values outside fp16 normal range: {scale_range}")
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file(progress=False)
        writer.close()
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
