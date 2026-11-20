#!/usr/bin/env python3
"""Reference numpy implementation of LittleBit packing and the factorised forward pass.

Written from the formulas and conventions documented in RESEARCH.md
(learned from refs/LittleBit/quantization/utils/binary_packer.py and
refs/LittleBit/quantization/modules/littlebit.py:118), not copied from them.

LittleBit packing: each row of a {-1,+1} matrix is split into 32-bit words, least
significant bit first, bit 0 = +1, bit 1 = -1, the last word padded with +1.

ggml Q1_0: blocks of 128 values, each block = fp16 scale d followed by 16 bytes,
least significant bit first, bit 1 = +d, bit 0 = -d.

Run with no arguments for the self-test. Pass the path to llama.cpp's gguf-py to also
compare against its Q1_0 writer:  reference_forward.py /path/to/llama.cpp/gguf-py
"""
import sys

import numpy as np

Q1_0_BLOCK = 128


def pack_signs(signs: np.ndarray) -> np.ndarray:
    """{-1,+1} matrix [rows, cols] -> int32 [rows, ceil(cols/32)] in LittleBit's convention."""
    rows, cols = signs.shape
    words = -(-cols // 32)
    bits = np.zeros((rows, words * 32), dtype=np.uint8)      # 0 = +1, which is also the padding value
    bits[:, :cols] = signs < 0
    return np.packbits(bits, axis=1, bitorder="little").view("<u4").astype(np.int32)


def unpack_signs(packed: np.ndarray, cols: int) -> np.ndarray:
    """Inverse of pack_signs: int32 [rows, words] -> float32 {-1,+1} [rows, cols]."""
    as_bytes = np.ascontiguousarray(packed.astype("<i4")).view(np.uint8)
    bits = np.unpackbits(as_bytes, axis=1, bitorder="little")[:, :cols]
    return np.where(bits == 1, -1.0, 1.0).astype(np.float32)


def pad_rank(u_packed, v_packed, mid, rank, multiple=Q1_0_BLOCK):
    """Pad the rank axis so U's rows are a whole number of Q1_0 blocks.

    U gains columns and V gains rows; both are filled with +1 (bit 0). `mid` (= v1*u2)
    is padded with zeros, so the added rank components contribute exactly nothing.
    """
    padded = -(-rank // multiple) * multiple
    u_bits = np.zeros((u_packed.shape[0], padded // 32), dtype=np.int32)
    u_bits[:, : u_packed.shape[1]] = u_packed
    v_bits = np.zeros((padded, v_packed.shape[1]), dtype=np.int32)
    v_bits[:rank] = v_packed
    mid_padded = np.zeros(padded, dtype=mid.dtype)
    mid_padded[:rank] = mid
    return u_bits, v_bits, mid_padded, padded


def littlebit_to_q1_0(packed: np.ndarray) -> np.ndarray:
    """LittleBit packed rows -> raw Q1_0 tensor bytes, without unpacking.

    Requires the row length to be a multiple of 128 (4 words per block). The sign bits are
    inverted (the two formats use opposite polarity) and each block gets d = 1.0.
    """
    rows, words = packed.shape
    assert words % 4 == 0, "row length must be a multiple of 128 values"
    qs = (~np.ascontiguousarray(packed.astype("<i4")).view(np.uint8)).reshape(rows, words // 4, 16)
    d = np.broadcast_to(np.array([1.0], dtype="<f2").view(np.uint8), (rows, words // 4, 2))
    return np.concatenate([d, qs], axis=-1).reshape(rows, -1)


def q1_0_to_dense(raw: np.ndarray) -> np.ndarray:
    """Raw Q1_0 bytes [rows, blocks*18] -> float32 [rows, blocks*128]."""
    rows = raw.shape[0]
    blocks = raw.reshape(rows, -1, 18)
    d = blocks[:, :, :2].copy().view("<f2").astype(np.float32)
    bits = np.unpackbits(blocks[:, :, 2:], axis=-1, bitorder="little")
    return np.where(bits == 1, d, -d).reshape(rows, -1)


def branch_forward(x, u, v, u1, mid, v2):
    """One branch: y = ((((x * v2) @ V^T) * mid) @ U^T) * u1, with mid = v1 * u2.

    x [n, in], V [rank, in], U [out, rank]; u1 [out], mid [rank], v2 [in].
    """
    return ((((x * v2) @ v.T) * mid) @ u.T) * u1


def littlebit_forward(x, main, residual=None):
    """`main` and `residual` are dicts with keys u, v, u1, mid, v2 (dense ±1 matrices)."""
    y = branch_forward(x, **main)
    if residual is not None:
        y = y + branch_forward(x, **residual)
    return y


def _random_branch(rng, out, inp, rank):
    return dict(
        u=rng.choice([-1.0, 1.0], size=(out, rank)).astype(np.float32),
        v=rng.choice([-1.0, 1.0], size=(rank, inp)).astype(np.float32),
        u1=rng.normal(size=out).astype(np.float32),
        mid=rng.normal(size=rank).astype(np.float32),
        v2=rng.normal(size=inp).astype(np.float32),
    )


def self_test(gguf_py: str | None = None) -> None:
    rng = np.random.default_rng(0)
    # Shapes of the real checkpoint's k_proj (out 1024, in 4096, rank 208) and a ragged case.
    for out, inp, rank in ((1024, 4096, 208), (96, 256, 40)):
        main, res = _random_branch(rng, out, inp, rank), _random_branch(rng, out, inp, rank)
        x = rng.normal(size=(3, inp)).astype(np.float32)

        # 1. pack -> unpack is the identity
        for b in (main, res):
            assert np.array_equal(unpack_signs(pack_signs(b["u"]), rank), b["u"])
            assert np.array_equal(unpack_signs(pack_signs(b["v"]), inp), b["v"])

        # 2. factorised forward equals the dense weight W = U diag(mid) V scaled by u1 and v2
        want = sum((b["u1"][:, None] * (b["u"] * b["mid"]) @ b["v"] * b["v2"][None, :]) for b in (main, res))
        got = littlebit_forward(x, main, res)
        err_dense = np.abs(got - x @ want.T).max() / np.abs(got).max()

        # 3. the Q1_0 route (pad rank to 128, convert packed bits directly) gives the same output
        via_q1 = []
        for b in (main, res):
            u_p, v_p, mid_p, padded = pad_rank(pack_signs(b["u"]), pack_signs(b["v"]), b["mid"], rank)
            u_q, v_q = littlebit_to_q1_0(u_p), littlebit_to_q1_0(v_p)
            assert u_q.shape == (out, padded // 128 * 18) and v_q.shape == (padded, inp // 128 * 18)
            via_q1.append(dict(u=q1_0_to_dense(u_q), v=q1_0_to_dense(v_q), u1=b["u1"], mid=mid_p, v2=b["v2"]))
            assert np.array_equal(via_q1[-1]["u"][:, :rank], b["u"]) and np.array_equal(via_q1[-1]["v"][:rank], b["v"])
        err_q1 = np.abs(littlebit_forward(x, *via_q1) - got).max() / np.abs(got).max()

        # 4. optional: our direct conversion is byte-identical to llama.cpp's own Q1_0 writer
        same_bytes = "skipped"
        if gguf_py:
            sys.path.insert(0, gguf_py)
            import gguf  # noqa: PLC0415
            ours = littlebit_to_q1_0(pad_rank(pack_signs(main["u"]), pack_signs(main["v"]), main["mid"], rank)[1])
            theirs = gguf.quants.quantize(via_q1[0]["v"], gguf.GGMLQuantizationType.Q1_0)
            same_bytes = bool(np.array_equal(ours, theirs.reshape(ours.shape)))
            assert same_bytes, "direct conversion differs from gguf-py Q1_0"

        print(f"out={out} in={inp} rank={rank}: max rel diff vs dense {err_dense:.2e}, "
              f"via Q1_0 {err_q1:.2e}, bytes equal gguf-py: {same_bytes}")
        assert err_dense < 1e-5 and err_q1 < 1e-5
    print("self-test passed")


if __name__ == "__main__":
    self_test(sys.argv[1] if len(sys.argv) > 1 else None)
