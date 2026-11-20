#!/usr/bin/env python3
"""Summarise a LittleBit checkpoint from its safetensors header only (no weights needed).

Usage: inspect_checkpoint.py [path/to/model.safetensors.header.json]
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

DEFAULT = Path(__file__).parent.parent / "work" / "checkpoint-meta" / "model.safetensors.header.json"
Q1_0_BLOCK = 128


def nbytes(entry):
    start, end = entry["data_offsets"]
    return end - start


def pad_to(n, multiple):
    return -(-n // multiple) * multiple


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    header = json.loads(path.read_text())
    header.pop("__metadata__", None)

    layer_re = re.compile(r"^model\.layers\.(\d+)\.(.+)\.V_packed$")
    rows = {}                      # projection name -> (out, in, rank)
    packed = defaultdict(int)      # projection name -> packed sign bytes over all layers and branches
    layers = set()
    for name, entry in header.items():
        m = layer_re.match(name)
        if not m:
            continue
        layer, proj = int(m.group(1)), m.group(2)
        layers.add(layer)
        base = name[: -len("V_packed")]
        rank, in_words = entry["shape"]
        out = header[base + "U_packed"]["shape"][0]
        dims = (out, in_words * 32, rank)
        assert rows.setdefault(proj, dims) == dims, f"{proj}: shape differs between layers"
        assert header[base + "u2"]["shape"] == [1, rank], f"{name}: u2 does not match rank"
        for suffix in ("U_packed", "V_packed", "U_R_packed", "V_R_packed"):
            if base + suffix in header:
                packed[proj] += nbytes(header[base + suffix])

    n_layers = len(layers)
    print(f"layers: {n_layers}")
    print(f"{'projection':22} {'out':>6} {'in':>6} {'rank':>5} {'rank->128':>9} {'bpw':>6} {'packed MB':>10}")
    total = 0
    for proj, (out, inp, rank) in sorted(rows.items()):
        mb = packed[proj] / 1e6
        total += packed[proj]
        # bits per weight of the sign matrices alone, two branches, as stored
        bpw = packed[proj] * 8 / (out * inp * n_layers)
        print(f"{proj:22} {out:6d} {inp:6d} {rank:5d} {pad_to(rank, Q1_0_BLOCK):9d} {bpw:6.3f} {mb:10.1f}")
    print(f"{'total packed signs':56} {total / 1e6:10.1f}")

    dense = {k: nbytes(v) for k, v in header.items() if "_packed" not in k}
    big = sorted(dense.items(), key=lambda kv: -kv[1])[:2]
    for name, size in big:
        print(f"{name:56} {size / 1e6:10.1f}")
    print(f"{'everything else (scales, norms, shapes)':56} {(sum(dense.values()) - sum(s for _, s in big)) / 1e6:10.1f}")


if __name__ == "__main__":
    main()
