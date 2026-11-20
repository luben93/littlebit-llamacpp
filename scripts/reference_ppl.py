#!/usr/bin/env python3
"""Independent float32 reference for a LittleBit Qwen3 checkpoint: perplexity with llama-perplexity's protocol.

Reads the original safetensors checkpoint (not the GGUF), runs a plain PyTorch Qwen3 forward where every
factorised projection is  y = ((((x*v2) @ V^T) * (v1*u2)) @ U^T) * u1  (+ residual path), and scores
windows the way tools/perplexity does: non-overlapping windows of n_ctx tokens, only the second half scored.

Sign matrices stay bit-packed in memory and are unpacked per use, so memory stays near the checkpoint size.

Usage: reference_ppl.py CHECKPOINT_DIR TEXT_FILE --ctx 2048 --chunks 4 [--threads 10]
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoTokenizer


def unpack(packed: torch.Tensor, cols: int) -> torch.Tensor:
    b = np.unpackbits(np.ascontiguousarray(packed.numpy().astype("<i4")).view(np.uint8), axis=1, bitorder="little")
    return torch.from_numpy(np.where(b[:, :cols] == 1, np.float32(-1), np.float32(1)))


class Model:
    def __init__(self, ckpt: Path):
        self.cfg = json.loads((ckpt / "config.json").read_text())
        self.st = safe_open(ckpt / "model.safetensors", "pt")
        self.names = set(self.st.keys())
        self.embd = self.st.get_tensor("model.embed_tokens.weight").float()
        self.head = self.st.get_tensor("lm_head.weight").float()

    def f(self, name):
        return self.st.get_tensor(name).float()

    def linear(self, base: str, x: torch.Tensor) -> torch.Tensor:
        y = None
        for s in ("", "_R"):
            if f"{base}.U{s}_packed" not in self.names:
                continue
            out_f, rank = (int(v) for v in self.st.get_tensor(f"{base}.U{s}_shape"))
            _, in_f = (int(v) for v in self.st.get_tensor(f"{base}.V{s}_shape"))
            u = unpack(self.st.get_tensor(f"{base}.U{s}_packed"), rank)
            v = unpack(self.st.get_tensor(f"{base}.V{s}_packed"), in_f)
            mid = self.f(f"{base}.v1{s}") * self.f(f"{base}.u2{s}")
            t = ((((x * self.f(f"{base}.v2{s}")) @ v.T) * mid) @ u.T) * self.f(f"{base}.u1{s}")
            y = t if y is None else y + t
        return y

    def rms(self, x, w):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.cfg["rms_norm_eps"]) * w

    @torch.no_grad()
    def logits(self, ids: torch.Tensor) -> torch.Tensor:
        c = self.cfg
        n, hd, nh, nkv = ids.shape[0], c["head_dim"], c["num_attention_heads"], c["num_key_value_heads"]
        inv = 1.0 / (c["rope_theta"] ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd))
        ang = torch.arange(n, dtype=torch.float32)[:, None] * inv[None, :]
        cos, sin = torch.cat([ang.cos(), ang.cos()], -1), torch.cat([ang.sin(), ang.sin()], -1)

        def rope(t):                                   # t [heads, n, hd]
            t1, t2 = t[..., : hd // 2], t[..., hd // 2:]
            return t * cos + torch.cat([-t2, t1], -1) * sin

        x = self.embd[ids]
        for i in range(c["num_hidden_layers"]):
            p = f"model.layers.{i}"
            h = self.rms(x, self.f(f"{p}.input_layernorm.weight"))
            q = self.linear(f"{p}.self_attn.q_proj", h).view(n, nh, hd).transpose(0, 1)
            k = self.linear(f"{p}.self_attn.k_proj", h).view(n, nkv, hd).transpose(0, 1)
            v = self.linear(f"{p}.self_attn.v_proj", h).view(n, nkv, hd).transpose(0, 1)
            q = rope(self.rms(q, self.f(f"{p}.self_attn.q_norm.weight")))
            k = rope(self.rms(k, self.f(f"{p}.self_attn.k_norm.weight")))
            k, v = k.repeat_interleave(nh // nkv, 0), v.repeat_interleave(nh // nkv, 0)
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(0, 1).reshape(n, nh * hd)
            x = x + self.linear(f"{p}.self_attn.o_proj", a)
            h = self.rms(x, self.f(f"{p}.post_attention_layernorm.weight"))
            g = F.silu(self.linear(f"{p}.mlp.gate_proj", h)) * self.linear(f"{p}.mlp.up_proj", h)
            x = x + self.linear(f"{p}.mlp.down_proj", g)
        return self.rms(x, self.f("model.norm.weight")) @ self.head.T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("text", type=Path)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--chunks", type=int, default=4)
    ap.add_argument("--threads", type=int, default=10)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    tok = AutoTokenizer.from_pretrained(args.checkpoint)
    ids = torch.tensor(tok(args.text.read_text(), add_special_tokens=False)["input_ids"])
    print(f"tokens: {len(ids)}, windows available: {len(ids) // args.ctx}")
    m = Model(args.checkpoint)
    first, nll, count = args.ctx // 2, 0.0, 0
    for c in range(args.chunks):
        t0 = time.time()
        w = ids[c * args.ctx:(c + 1) * args.ctx]
        lp = F.log_softmax(m.logits(w)[first:-1].double(), -1)
        nll += -lp.gather(1, w[first + 1:, None]).sum().item()
        count += args.ctx - first - 1
        print(f"[{c + 1}] ppl so far {math.exp(nll / count):.4f}  ({time.time() - t0:.0f}s)", flush=True)
    print(f"reference perplexity over {args.chunks} windows: {math.exp(nll / count):.4f}")


if __name__ == "__main__":
    main()
