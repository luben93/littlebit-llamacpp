#!/usr/bin/env python3
"""Export blocks trained by layerwise_littlebit.py / finetune_scales.py as a LittleBit checkpoint.

The output directory has the same tensor names and packing as checkpoints written by the
SamsungLabs/LittleBit code (`<proj>.U_packed`, `V_packed`, `U_shape`, `V_shape`, `u1`, `u2`, `v1`, `v2`
and the `_R` residual variants), so any LittleBit -> GGUF converter can read it.

Our blocks store one scale vector `mid` per path where LittleBit has `v1` and `u2`; only their product
is used in the forward pass, so `v1 = mid` and `u2 = 1`.

Usage:
  export_layerwise_checkpoint.py ORIGINAL_MODEL_DIR BLOCKS_DIR OUT_DIR
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

PROJ = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]


def pack(signs01: torch.Tensor) -> torch.Tensor:
    """int8 matrix with 1 = +1 and 0 = -1  ->  int32 words, lsb first, bit 1 = -1, row padding = +1."""
    rows, cols = signs01.shape
    words = -(-cols // 32)
    bits = np.zeros((rows, words * 32), dtype=np.uint8)
    bits[:, :cols] = signs01.numpy() == 0
    return torch.from_numpy(np.packbits(bits, axis=1, bitorder="little").view("<u4").astype(np.int32))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=Path, help="original Hugging Face model directory")
    ap.add_argument("blocks", type=Path, help="directory with block_NNN.safetensors")
    ap.add_argument("out", type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    out, bits, weights, residual = {}, 0, 0, False
    with safe_open(args.model / "model.safetensors", "pt") as st:
        proj_weights = {f"model.layers.{i}.{p}.weight" for i in range(1000) for p in PROJ}
        for name in st.keys():
            if name not in proj_weights:
                out[name] = st.get_tensor(name)
        # Some checkpoints store the tied output head as a second copy of the embedding; keep one.
        tied = json.loads((args.model / "config.json").read_text()).get("tie_word_embeddings", False)
        if tied and "lm_head.weight" in out and torch.equal(out["lm_head.weight"], out["model.embed_tokens.weight"]):
            del out["lm_head.weight"]
            print("dropped lm_head.weight (identical to the tied embedding)")
        n_layer = 1 + max(int(n.split(".")[2]) for n in st.keys() if n.startswith("model.layers."))

        for i in range(n_layer):
            blk = load_file(args.blocks / f"block_{i:03d}.safetensors")
            for p in PROJ:
                base = f"model.layers.{i}.{p}"
                out_f, in_f = st.get_slice(f"{base}.weight").get_shape()
                weights += out_f * in_f
                for j, sfx in enumerate(("", "_R")):
                    if f"{p}.{j}.U" not in blk:
                        continue
                    residual |= j == 1
                    u, v = blk[f"{p}.{j}.U"], blk[f"{p}.{j}.V"]
                    rank = v.shape[0]
                    assert u.shape == (out_f, rank) and v.shape == (rank, in_f), (base, u.shape, v.shape)
                    out[f"{base}.U{sfx}_packed"] = pack(u)
                    out[f"{base}.V{sfx}_packed"] = pack(v)
                    out[f"{base}.U{sfx}_shape"] = torch.tensor([out_f, rank], dtype=torch.int64)
                    out[f"{base}.V{sfx}_shape"] = torch.tensor([rank, in_f], dtype=torch.int64)
                    out[f"{base}.u1{sfx}"] = blk[f"{p}.{j}.u1"].float().reshape(1, out_f)
                    out[f"{base}.u2{sfx}"] = torch.ones(1, rank)
                    out[f"{base}.v1{sfx}"] = blk[f"{p}.{j}.mid"].float().reshape(1, rank)
                    out[f"{base}.v2{sfx}"] = blk[f"{p}.{j}.v2"].float().reshape(1, in_f)
                    bits += rank * (out_f + in_f) + 16 * (out_f + rank + in_f)   # scales counted at 16 bits

    eff = bits / weights
    save_file(out, args.out / "model.safetensors", metadata={"format": "pt"})
    for f in args.model.iterdir():
        if f.is_file() and f.name != "model.safetensors":
            shutil.copy2(f, args.out / f.name)
    (args.out / "littlebit_config.json").write_text(json.dumps({
        "quant_func": "SmoothSign", "quant_mod": "LittleBitLinear", "eff_bit": round(eff, 4),
        "residual": residual, "use_itq": False, "kv_factor": 1.0, "min_split_dim": 8,
        "note": "layer-wise trained (scripts/layerwise_littlebit.py + finetune_scales.py), not the papers' QAT; "
                "ranks are multiples of 128 and do not follow the LittleBit rank formula",
    }, indent=2) + "\n")
    print(f"{n_layer} layers, residual={residual}, {eff:.3f} bits per compressed weight "
          f"({bits / 8e6:.1f} MB incl. scales) -> {args.out}")


if __name__ == "__main__":
    main()
