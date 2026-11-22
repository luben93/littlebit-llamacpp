#!/usr/bin/env python3
"""Layer-wise LittleBit compression of a Qwen3 model on small hardware (experimental).

The LittleBit papers train the whole compressed model end to end against the original (QAT + distillation).
This script instead trains one transformer block at a time: the compressed block is fitted to reproduce
the original block's output on calibration text, then its output feeds the next block. Only one block is
trained at a time, so it fits a small GPU. This is NOT the papers' training recipe and gives lower quality.

Each linear layer W [out, in] becomes, per path (primary and residual):
    y = u1 * (sign(U) @ (mid * (sign(V) @ (v2 * x))))
initialised from a truncated SVD of W (signs of the factors, rank-1 fits of their magnitudes), as the
papers describe (Dual-SVID). Ranks are chosen as multiples of 128 so the GGUF needs no padding.

Usage:
  layerwise_littlebit.py MODEL_DIR --train-text wiki.train.raw --test-text wiki.test.raw \
      --bpw 1.0 --steps 300 [--out factors.safetensors]
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoTokenizer

PROJ = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]


class SignSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.where(x >= 0, 1.0, -1.0).to(x.dtype)

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        return g * (x.abs() <= 1.0).to(g.dtype)


def rank_for(out_f: int, in_f: int, bpw: float, residual: bool) -> int:
    """Largest multiple of 128 whose stored size stays at or below `bpw` (signs 1 bit, scales 16 bits)."""
    paths = 2 if residual else 1
    r = (out_f * in_f * bpw - 16 * paths * (out_f + in_f)) / (paths * (out_f + in_f + 16))
    return max(128, int(r) // 128 * 128)


def svid_init(w: torch.Tensor, r: int):
    """One path: signs of the SVD factors, rank-1 magnitude fits as scales. Returns params and the approximation."""
    u, s, vh = torch.linalg.svd(w, full_matrices=False)
    uf = u[:, :r] * s[:r].sqrt()                      # [out, r]
    vf = s[:r].sqrt()[:, None] * vh[:r]               # [r, in]

    def rank1(a):                                     # |a| ~= outer(p, q)
        pu, ps, pv = torch.svd_lowrank(a.abs(), q=1, niter=4)
        return (pu[:, 0] * ps[0].sqrt()).abs(), (pv[:, 0] * ps[0].sqrt()).abs()

    u1, u2 = rank1(uf)                                # [out], [r]
    v1, v2 = rank1(vf)                                # [r], [in]
    # latents are scaled so their typical magnitude is below 1 (inside the STE window)
    p = dict(U=uf / (uf.abs().max() + 1e-8), V=vf / (vf.abs().max() + 1e-8), u1=u1, mid=u2 * v1, v2=v2)
    approx = (u1[:, None] * torch.sign(uf)) * p["mid"][None, :] @ (torch.sign(vf) * v2[None, :])
    return p, approx


class LBLinear(torch.nn.Module):
    def __init__(self, w: torch.Tensor, bpw: float, residual: bool):
        super().__init__()
        out_f, in_f = w.shape
        self.rank = min(rank_for(out_f, in_f, bpw, residual), min(out_f, in_f) // 128 * 128)
        self.paths = torch.nn.ModuleList()
        target = w
        for _ in range(2 if residual else 1):
            p, approx = svid_init(target, self.rank)
            self.paths.append(torch.nn.ParameterDict({k: torch.nn.Parameter(v.clone()) for k, v in p.items()}))
            target = target - approx

    def forward(self, x):
        y = 0
        for p in self.paths:
            t = (x * p["v2"]) @ SignSTE.apply(p["V"]).T
            y = y + ((t * p["mid"]) @ SignSTE.apply(p["U"]).T) * p["u1"]
        return y

    def bits(self):
        return sum(p["U"].numel() + p["V"].numel() + 16 * (p["u1"].numel() + p["mid"].numel() + p["v2"].numel())
                   for p in self.paths)


class Qwen3:
    """Minimal Qwen3 forward; each block's linears are either dense weights or LBLinear modules."""

    def __init__(self, model_dir: Path, device: str):
        self.cfg = json.loads((model_dir / "config.json").read_text())
        self.dev = device
        self.st = safe_open(model_dir / "model.safetensors", "pt")
        self.embd = self.t("model.embed_tokens.weight")
        self.head = self.t("lm_head.weight") if "lm_head.weight" in self.st.keys() else self.embd
        self.n_layer = self.cfg["num_hidden_layers"]

    def t(self, name):
        return self.st.get_tensor(name).float().to(self.dev)

    def block_weights(self, i):
        p = f"model.layers.{i}"
        w = {k: self.t(f"{p}.{k}.weight") for k in PROJ}
        for k in ("input_layernorm", "post_attention_layernorm", "self_attn.q_norm", "self_attn.k_norm"):
            w[k] = self.t(f"{p}.{k}.weight")
        return w

    def rms(self, x, w):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.cfg["rms_norm_eps"]) * w

    def rope(self, n):
        hd = self.cfg["head_dim"]
        inv = 1.0 / (self.cfg["rope_theta"] ** (torch.arange(0, hd, 2, dtype=torch.float32, device=self.dev) / hd))
        ang = torch.arange(n, dtype=torch.float32, device=self.dev)[:, None] * inv[None, :]
        return torch.cat([ang.cos(), ang.cos()], -1), torch.cat([ang.sin(), ang.sin()], -1)

    def block(self, x, w, lin=None):
        """x [batch, n, hidden]; w dense weights and norms; lin optional dict name -> LBLinear."""
        c = self.cfg
        b, n, _ = x.shape
        hd, nh, nkv = c["head_dim"], c["num_attention_heads"], c["num_key_value_heads"]
        cos, sin = self.rope(n)

        def L(name, v):
            return lin[name](v) if lin is not None else v @ w[name].T

        def rot(t):
            return t * cos + torch.cat([-t[..., hd // 2:], t[..., : hd // 2]], -1) * sin

        h = self.rms(x, w["input_layernorm"])
        q = L("self_attn.q_proj", h).view(b, n, nh, hd).transpose(1, 2)
        k = L("self_attn.k_proj", h).view(b, n, nkv, hd).transpose(1, 2)
        v = L("self_attn.v_proj", h).view(b, n, nkv, hd).transpose(1, 2)
        q, k = rot(self.rms(q, w["self_attn.q_norm"])), rot(self.rms(k, w["self_attn.k_norm"]))
        k, v = k.repeat_interleave(nh // nkv, 1), v.repeat_interleave(nh // nkv, 1)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(b, n, nh * hd)
        x = x + L("self_attn.o_proj", a)
        h = self.rms(x, w["post_attention_layernorm"])
        return x + L("mlp.down_proj", F.silu(L("mlp.gate_proj", h)) * L("mlp.up_proj", h))

    def logits(self, x):
        return self.rms(x, self.t("model.norm.weight")) @ self.head.T


@torch.no_grad()
def perplexity(model, lins, ids, ctx, chunks):
    """lins: per-block dict name -> LBLinear kept on the CPU, or None for the dense teacher.

    llama-perplexity protocol: second half of each window scored. Blocks visit the GPU one at a time.
    """
    first = ctx // 2
    n = min(chunks, len(ids) // ctx)
    win = [ids[c * ctx:(c + 1) * ctx] for c in range(n)]
    xs = [model.embd[w.to(model.dev)][None].half().cpu() for w in win]
    for i in range(model.n_layer):
        w = model.block_weights(i)
        lin = None
        if lins is not None:
            lin = {k: m.to(model.dev).float() for k, m in lins[i].items()}
        xs = [model.block(x.to(model.dev).float(), w, lin).half().cpu() for x in xs]
        if lin is not None:
            for m in lin.values():
                m.half().cpu()
        del w
    nll, count = 0.0, 0
    for x, w_ids in zip(xs, win):
        w_ids = w_ids.to(model.dev)
        for s in range(first, ctx - 1, 256):                         # logits in slices: vocab is large
            e = min(s + 256, ctx - 1)
            lp = F.log_softmax(model.logits(x[:, s:e].to(model.dev).float())[0].double(), -1)
            nll += -lp.gather(1, w_ids[s + 1:e + 1, None]).sum().item()
        count += ctx - first - 1
    return math.exp(nll / count)


def rel_mse(out, target):
    """Mean over tokens of |out - target|^2 / |target|^2.

    Plain MSE is dominated by the few tokens with very large activations (the first position of a
    sequence acts as an attention sink in Qwen models), so every token is weighted equally instead.
    """
    return ((out - target).pow(2).sum(-1) / (target.pow(2).sum(-1) + 1e-6)).mean()


def save_block(path, view):
    from safetensors.torch import save_file
    d = {}
    for name, mod in view.items():
        for j, pth in enumerate(mod.paths):
            for k, v in pth.items():
                d[f"{name}.{j}.{k}"] = ((v.detach() >= 0).to(torch.int8) if k in ("U", "V") else v.detach().float()).contiguous().cpu()
    save_file(d, path)


def load_block(path, w, bpw, residual):
    """Rebuild LBLinear modules from a saved block without redoing the SVD."""
    from safetensors.torch import load_file
    d = load_file(path)
    view = {}
    for name in PROJ:
        mod = LBLinear.__new__(LBLinear)
        torch.nn.Module.__init__(mod)
        mod.paths = torch.nn.ModuleList()
        j = 0
        while f"{name}.{j}.U" in d:
            pd = {k: d[f"{name}.{j}.{k}"] for k in ("U", "V", "u1", "mid", "v2")}
            pd["U"], pd["V"] = pd["U"].float() * 2 - 1, pd["V"].float() * 2 - 1
            mod.paths.append(torch.nn.ParameterDict({k: torch.nn.Parameter(v, requires_grad=False) for k, v in pd.items()}))
            j += 1
        mod.rank = mod.paths[0]["mid"].numel()
        view[name] = mod.half()
    return view


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=Path)
    ap.add_argument("--train-text", type=Path, required=True)
    ap.add_argument("--test-text", type=Path, required=True)
    ap.add_argument("--bpw", type=float, default=1.0)
    ap.add_argument("--no-residual", action="store_true")
    ap.add_argument("--steps", type=int, default=300, help="training steps per block (0 = initialisation only)")
    ap.add_argument("--seqs", type=int, default=256, help="calibration sequences")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-ctx", type=int, default=2048)
    ap.add_argument("--eval-chunks", type=int, default=12)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, required=True, help="directory for per-block results; rerun to resume")
    ap.add_argument("--loss", default="rel", choices=["rel", "mse"])
    args = ap.parse_args()
    torch.manual_seed(0)
    residual = not args.no_residual
    args.out.mkdir(parents=True, exist_ok=True)
    loss_fn = rel_mse if args.loss == "rel" else F.mse_loss

    tok = AutoTokenizer.from_pretrained(args.model)
    enc = lambda p: torch.tensor(tok(p.read_text(), add_special_tokens=False)["input_ids"])  # noqa: E731
    train_ids, test_ids = enc(args.train_text), enc(args.test_text)
    m = Qwen3(args.model, args.device)
    n_cal = min(args.seqs, len(train_ids) // args.seq_len)
    cal = train_ids[: n_cal * args.seq_len].view(n_cal, args.seq_len)
    print(f"device {args.device}; calibration {n_cal} x {args.seq_len} tokens; test tokens {len(test_ids)}")

    with torch.no_grad():
        x_teacher = m.embd[cal.to(args.device)].half().cpu()       # teacher stream
    x_student = x_teacher.clone()                                   # student stream
    blocks, total_bits, total_weights, t_start = [], 0, 0, time.time()

    for i in range(m.n_layer):
        t0 = time.time()
        w = m.block_weights(i)
        saved = args.out / f"block_{i:03d}.safetensors"
        done = saved.exists()
        if done:
            view = {k: m_.to(args.device).float() for k, m_ in load_block(saved, w, args.bpw, residual).items()}
            lin = torch.nn.ModuleDict({k.replace(".", "__"): v for k, v in view.items()})
        else:
            lin = torch.nn.ModuleDict({k.replace(".", "__"): LBLinear(w[k], args.bpw, residual) for k in PROJ}).to(args.device)
            view = {k: lin[k.replace(".", "__")] for k in PROJ}
        total_bits += sum(l.bits() for l in view.values())
        total_weights += sum(w[k].numel() for k in PROJ)

        with torch.no_grad():                                        # teacher targets for this block
            y_teacher = torch.cat([m.block(x_teacher[j:j + args.batch].to(args.device).float(), w).half().cpu()
                                   for j in range(0, n_cal, args.batch)])

        def mse():
            with torch.no_grad():
                e = [loss_fn(m.block(x_student[j:j + args.batch].to(args.device).float(), w, view),
                             y_teacher[j:j + args.batch].to(args.device).float()).item()
                     for j in range(0, min(n_cal, 4 * args.batch), args.batch)]
            return sum(e) / len(e)

        loss0 = mse()
        if args.steps and not done:
            scales = [p for n, p in lin.named_parameters() if n.split(".")[-1] in ("u1", "mid", "v2")]
            latents = [p for n, p in lin.named_parameters() if n.split(".")[-1] in ("U", "V")]
            opt = torch.optim.Adam([{"params": scales, "lr": args.lr}, {"params": latents, "lr": args.lr * 0.1}])
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[args.lr, args.lr * 0.1], total_steps=args.steps)
            for _ in range(args.steps):
                idx = torch.randint(0, n_cal, (args.batch,))
                out = m.block(x_student[idx].to(args.device).float(), w, view)
                loss = loss_fn(out, y_teacher[idx].to(args.device).float())
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
        loss1 = mse()

        with torch.no_grad():                                        # advance both streams
            x_student = torch.cat([m.block(x_student[j:j + args.batch].to(args.device).float(), w, view).half().cpu()
                                   for j in range(0, n_cal, args.batch)])
        x_teacher = y_teacher
        for p in lin.parameters():
            p.requires_grad_(False)
        if not done:
            save_block(saved, view)
        for mod in view.values():                                   # keep only signs for U and V, park on the CPU
            for path in mod.paths:
                path["U"].copy_(torch.where(path["U"] >= 0, 1.0, -1.0))
                path["V"].copy_(torch.where(path["V"] >= 0, 1.0, -1.0))
            mod.half().cpu()
        blocks.append(view)
        del w, lin
        torch.cuda.empty_cache()
        print(f"block {i:2d}: ranks q/k/mlp {view['self_attn.q_proj'].rank}/{view['self_attn.k_proj'].rank}/"
              f"{view['mlp.up_proj'].rank}  block {args.loss} {loss0:.4g} -> {loss1:.4g}  ({time.time() - t0:.0f}s){" [loaded]" if done else ""}", flush=True)

    print(f"compressed layers: {total_bits / total_weights:.3f} bits per weight, {total_bits / 8e6:.0f} MB; "
          f"total time {time.time() - t_start:.0f}s")
    del x_teacher, x_student, y_teacher
    torch.cuda.empty_cache()
    print(f"teacher perplexity ({args.eval_chunks} windows): {perplexity(m, None, test_ids, args.eval_ctx, args.eval_chunks):.2f}")
    print(f"student perplexity ({args.eval_chunks} windows): {perplexity(m, blocks, test_ids, args.eval_ctx, args.eval_chunks):.2f}")

if __name__ == "__main__":
    main()
