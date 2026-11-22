# Layer-wise LittleBit compression on a laptop GPU: what we trained and how it went

This is an experiment report, not a recommendation. The method here is **not** the LittleBit papers' recipe.
The papers train the compressed model end to end against the original, which needs the whole model trainable in
GPU memory. We tried a cheaper route that fits a 6 GB laptop GPU, measured it against standard llama.cpp
quantizations of the same model with the same tool, and report everything, including what did not work.

## Result in one table

Qwen3-0.6B. All rows scored by `llama-perplexity` (patched llama.cpp, CUDA), windows of 2048 tokens, second half
of each window scored, whole file: WikiText-2 test (145 windows) and C4 validation text (138 windows).
Rows are sorted by file size. Raw logs and a machine-readable summary: `results/pilot-0.6b/`.

| Model | File (MB) | WikiText-2 PPL | C4 PPL |
|---|---:|---:|---:|
| Standard `Q1_0` (1-bit, no training) | 196 | 4,205,757 | 3,296,567 |
| **Ours, 1.08 bpw layers, Q4_K_M embeddings** | **200** | **52.52** | **194.5** |
| Standard `IQ1_S` | 208 | 6,398 | 6,131 |
| Standard `IQ1_M` | 216 | 1,514 | 1,671 |
| Standard `TQ1_0` | 227 | not finished (about 27 million after 18 windows) | not finished |
| Standard `IQ2_XXS` | 229 | 798.8 | 884.6 |
| **Ours, 1.08 bpw layers, Q8_0 embeddings** | **238** | **52.38** | **194.3** |
| Standard `Q2_K` | 296 | 34.37 | 49.13 |
| Standard `Q3_K_S` | 323 | 23.97 | 35.43 |
| **Ours, 1.08 bpw layers, F16 embeddings** | **384** | **52.38** | **194.3** |
| Standard `Q4_K_M` | 397 | 18.47 | 28.38 |
| Standard `Q8_0` | 639 | 17.90 | 27.22 |
| Original, F16 | 1198 | 17.96 | 27.27 |

The standard quantizations were made with `llama-quantize` and an importance matrix computed from the same text,
and the same amount of it, that our model was calibrated on (1024 × 512 tokens of WikiText-2 train).
The original Hugging Face file stores the tied output head as a second copy of the embedding; it was dropped
for every row, so sizes are comparable.

## What the numbers say

**Where it is better**

- Below about 240 MB it beats every standard quantization by a wide margin: 52.5 against 799 for `IQ2_XXS`
  (229 MB) on WikiText-2, and 194.5 against 885 on C4. The standard types at this size are broken for this model.
- Quantizing the embeddings from F16 to Q4_K_M costs almost nothing (52.38 to 52.52) and halves the file.
- The trained model runs in llama.cpp with the existing `Q1_0` kernels, and llama.cpp reproduces the PyTorch
  perplexity of the trained model: 44.36 against 44.37 on the first 12 windows.

**Where it is worse**

- `Q2_K` is 48% larger (296 MB against 200 MB) and much better: 34.4 against 52.5 on WikiText-2, and 49.1 against
  194.5 on C4. For a model this small, 96 MB more buys a far better model.
- It does not generalise. Relative to the original, perplexity is 2.9 times higher on WikiText-2 (the dataset it
  was calibrated on, different split) and 7.1 times higher on C4. The standard quantizations have about the same
  ratio on both texts (`Q2_K`: 1.9 and 1.8; `Q3_K_S`: 1.3 and 1.3). The model has partly fitted Wikipedia
  rather than the original model.
- It is behind end-to-end training. [littlebit-bench](https://github.com/christopherrobertbrooks-tech/littlebit-bench)
  reports 37.64 on WikiText-2 and 44.16 on C4 for the papers' recipe on the same model at 1.02 bits per weight
  (its own protocol and data, so not directly comparable). Its C4 figure is 1.2 times its WikiText-2 figure;
  ours is 3.7 times.
- At this model size the compressed layers are a small part of the file: 64 MB, with the embedding taking most
  of the rest. The method cannot make a 0.6B model much smaller than a standard 2-bit quantization already is.

**A number to be careful with.** The training scripts print perplexity on the first 12 windows of the test set
(44.4 for the final model). Over the whole test set it is 52.4. Use the table above.

**Not measured:** speed of the 0.6B files, any task benchmark, more than one seed, more than one model.

## Method

`scripts/layerwise_littlebit.py`, then `scripts/finetune_scales.py`.

1. **Block by block.** Each transformer block's seven projections are replaced by LittleBit factors
   (two paths, each `W ≈ U diag(m) V` with sign matrices `U`, `V`; SVD initialisation as in the papers). The signs
   and scales of that block are trained to reproduce the original block's output on calibration text. Blocks are
   done in order, and each block is fed the output of the already-compressed blocks before it.
2. **Global scale pass.** All sign matrices are frozen. Only the scale vectors of all blocks (1.40 M values) are
   trained together so that the output distribution matches the original model's (KL divergence on the logits).
   This is the only stage that sees the whole model.

Ranks are multiples of 128 (384 for q/o/MLP, 256 for k/v), so the GGUF needs no rank padding.
Compressed layers: 1.084 bits per weight, counting scales at 16 bits each.

## Hardware and cost

- Laptop: Intel Core i7-13700H, 30 GiB usable RAM, NVIDIA RTX A1000 6 GB, 150 W power adapter.
  PyTorch 2.6.0 with CUDA 12.4.
- GPU memory in use during block training: about 4.5 GB of 6 GB.
- GPU power during training: 48–50 W at 99% utilisation (`nvidia-smi`, two spot checks). Whole-machine draw was
  not measured; the adapter limits it to 150 W, so the final 7-hour run used at most about 1 kWh.

## Runs, in order, including the ones that failed

Perplexity in this table is the training scripts' own PyTorch evaluation on the first 12 test windows.

| # | Setting | Time | Outcome |
|---|---|---|---|
| 0 | Initialisation only, no training, 0.72 bpw | 2 min | About 86 million. Log not kept. |
| 1 | Block by block, plain MSE loss, 400 steps per block, 0.72 bpw | 35 min | **Lost.** The loss was dominated by a few tokens with very large activations (block MSE jumped from 0.05 to 86.6 at block 2). Then the evaluation ran out of GPU memory, and nothing had been saved. |
| 2 | Same, with squared error relative to each token's norm, plus per-block saving and resume | 35 min | 529.4 |
| 3 | 1.08 bpw, 1200 steps per block, 512 calibration sequences | 2 h | 144.4 |
| 4 | 1.08 bpw, 4000 steps per block, 1024 calibration sequences | 6.5 h | 81.8 |
| 5 | Run 4 + global scale pass, 100 steps | 51 s | 53.3. Log not kept. |
| 6 | Run 4 + global scale pass, 3000 steps | 25 min | 44.4 (**52.4 on the full test set, 194 on C4**) |

Observations:

- The 25-minute scale pass gained as much as run 4 did over run 3, which cost 4.5 hours more.
- The scale pass had not converged: its loss was still falling at step 3000 (1.19 to 0.91 KL per token).
- Run 4 used more than three times the steps and twice the calibration text of run 3. Every block's error fell
  by about a fifth (block 0: 0.194 to 0.152) and perplexity fell from 144 to 82.

Logs: `results/pilot-0.6b/train/`.

## What we would try next, and what we do not claim

- Calibrate on mixed text instead of WikiText-2 alone. The C4 result says this matters more than more steps.
- Run the scale pass longer; it is cheap and had not converged.
- We have not shown that this scales to larger models. The global pass keeps the whole compressed model in
  memory, and block time grows with block size. The stated case for sub-1-bit methods is large models, and that
  is exactly the part this experiment does not cover.

## Reproduce

`scripts/run_pilot_baselines.sh` is the full sequence behind the table: export of the trained blocks to a
LittleBit checkpoint (`scripts/export_layerwise_checkpoint.py`; same tensor names and packing as the Samsung code
writes, so other LittleBit converters can read it), conversion to GGUF, the importance matrix and the standard
quantizations, and every perplexity run. Training itself:

```sh
python scripts/layerwise_littlebit.py pilot/qwen3-0.6b --train-text data/wiki.train.raw \
    --test-text data/wiki.test.raw --bpw 1.2 --steps 4000 --seqs 1024 --eval-chunks 12 \
    --out pilot/run-rel-b1.2-s4000
python scripts/finetune_scales.py pilot/qwen3-0.6b --blocks pilot/run-rel-b1.2-s4000 \
    --train-text data/wiki.train.raw --test-text data/wiki.test.raw --steps 3000 \
    --out pilot/run3-scales-s3000
```

The trained weights are not in this repository (binary files are not tracked).
