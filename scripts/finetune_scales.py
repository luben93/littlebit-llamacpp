#!/usr/bin/env python3
"""Global scale-only fine-tune for a layer-wise LittleBit model (experimental).

Takes the per-block results of layerwise_littlebit.py, freezes every sign matrix, and trains only the
scale vectors (u1, mid, v2) of all blocks together so the compressed model's output distribution matches
the original model's (KL divergence on the logits). The scales are a tiny fraction of the parameters,
so this is the cheapest way to let later layers compensate for errors made in earlier ones.

Usage:
  finetune_scales.py MODEL_DIR --blocks RUN_DIR --train-text wiki.train.raw --test-text wiki.test.raw \
      --steps 1500 --out RUN_DIR_SCALES
"""
import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from layerwise_littlebit import PROJ, Qwen3, load_block, perplexity, save_block  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=Path)
    ap.add_argument("--blocks", type=Path, required=True)
    ap.add_argument("--train-text", type=Path, required=True)
    ap.add_argument("--test-text", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--eval-ctx", type=int, default=2048)
    ap.add_argument("--eval-chunks", type=int, default=12)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    torch.manual_seed(0)
    args.out.mkdir(parents=True, exist_ok=True)
    dev = args.device

    tok = AutoTokenizer.from_pretrained(args.model)
    enc = lambda p: torch.tensor(tok(p.read_text(), add_special_tokens=False)["input_ids"])  # noqa: E731
    train_ids, test_ids = enc(args.train_text), enc(args.test_text)
    m = Qwen3(args.model, dev)
    n_seq = len(train_ids) // args.seq_len
    seqs = train_ids[: n_seq * args.seq_len].view(n_seq, args.seq_len)

    # teacher weights (fp16) and student blocks (signs fp16, scales fp32 trainable) stay on the device
    teacher = []
    for i in range(m.n_layer):
        teacher.append({k: v.half() for k, v in m.block_weights(i).items()})
    student, scales = [], []
    for i in range(m.n_layer):
        view = load_block(args.blocks / f"block_{i:03d}.safetensors", None, None, None)
        for mod in view.values():
            mod.to(dev)
            for pth in mod.paths:
                for k in ("u1", "mid", "v2"):
                    pth[k].data = pth[k].data.float()
                    pth[k].requires_grad_(True)
                    scales.append(pth[k])
        student.append(view)
    n_scale = sum(p.numel() for p in scales)
    print(f"device {dev}; trainable scale values {n_scale / 1e6:.2f} M; sequences {n_seq} x {args.seq_len}")

    class Lin:
        """LittleBit linear with fp16 signs and fp32 scales (signs are constants here)."""

        def __init__(self, mod):
            self.mod = mod

        def __call__(self, x):
            y = 0
            for p in self.mod.paths:
                t = ((x * p["v2"]).half() @ p["V"].T).float()
                y = y + ((t * p["mid"]).half() @ p["U"].T).float() * p["u1"]
            return y

    def student_block(i, x):
        w = {k: v.float() for k, v in teacher[i].items() if k not in PROJ}
        return m.block(x, w, {k: Lin(student[i][k]) for k in PROJ})

    def run_student(x):
        # recompute activations in the backward pass instead of storing them for all blocks
        x = x.requires_grad_(True)
        for i in range(m.n_layer):
            x = torch.utils.checkpoint.checkpoint(student_block, i, x, use_reentrant=False)
        return x

    @torch.no_grad()
    def run_teacher(x):
        for i in range(m.n_layer):
            x = m.block(x, {k: v.float() for k, v in teacher[i].items()})
        return x

    opt = torch.optim.Adam(scales, lr=args.lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps)
    t0, run = time.time(), 0.0
    for step in range(1, args.steps + 1):
        ids = seqs[torch.randint(0, n_seq, (args.batch,))].to(dev)
        x0 = m.embd[ids]
        with torch.no_grad():
            t_logp = F.log_softmax(m.logits(run_teacher(x0)), -1)
        s_logp = F.log_softmax(m.logits(run_student(x0)), -1)
        loss = F.kl_div(s_logp, t_logp, log_target=True, reduction="batchmean") / args.seq_len
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(scales, 1.0)
        opt.step()
        sched.step()
        run += loss.item()
        if step % 100 == 0:
            print(f"step {step:5d}  KL per token {run / 100:.4f}  ({time.time() - t0:.0f}s)", flush=True)
            run = 0.0

    lins = []
    for i in range(m.n_layer):
        for mod in student[i].values():
            for p in mod.parameters():
                p.requires_grad_(False)
        save_block(args.out / f"block_{i:03d}.safetensors", student[i])
        lins.append({k: mod.half().cpu() for k, mod in student[i].items()})
    del teacher, student
    torch.cuda.empty_cache()
    print(f"student perplexity after scale fine-tune ({args.eval_chunks} windows): "
          f"{perplexity(m, lins, test_ids, args.eval_ctx, args.eval_chunks):.2f}")


if __name__ == "__main__":
    main()
