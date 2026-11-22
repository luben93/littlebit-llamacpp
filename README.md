# littlebit-llamacpp

A small loader patch and a converter that let llama.cpp run LittleBit / LittleBit-2 sub-1-bit checkpoints,
plus measurements on one published checkpoint.

**Whose work this is.** The compression method is LittleBit and LittleBit-2 by Banseok Lee, Dongkyu Kim,
Youngcheon You and Youngmin Kim at Samsung Research. They developed the method, the training and the
checkpoint format. This repository contributes only plumbing and tests: 64 lines that load their factorised
layers into llama.cpp's existing 1-bit `Q1_0` type, a script that converts their checkpoint files, and
measurements. No new tensor type and no new kernel. Please cite their papers (see [Citation](#citation)).

Research prototype. Not affiliated with or endorsed by Samsung or the llama.cpp project.

**Related projects.** [GeniusDevam/littlebit-gguf](https://github.com/GeniusDevam/littlebit-gguf) implements the same
approach independently, with validation, tests and CI for Llama, Qwen2 and Qwen3.
[ScarionWorkspace/littlebit-gguf](https://github.com/ScarionWorkspace/littlebit-gguf) is a third implementation that reports a
fused GPU path. The three overlap and are being compared in GeniusDevam's issue #1. This repository adds an
independent reference, measurements, and cross-checks between the builds
(see [Cross-checks](#cross-checks-with-geniusdevamlittlebit-gguf)).
Upstream discussion: [ggml-org/llama.cpp#32671](https://github.com/ggml-org/llama.cpp/discussions/32671).

The code and measurements here were produced with AI assistance (Claude Code).

## What is here

| Path | What it is |
|---|---|
| `scripts/convert_littlebit_to_gguf.py` | LittleBit safetensors checkpoint → GGUF |
| `patches/littlebit-q1_0.diff` | llama.cpp patch (loader + graph), against commit `9183a55` (2026-11-14) |
| `scripts/reference_ppl.py` | independent float32 PyTorch reference, scores perplexity the way `llama-perplexity` does |
| `scripts/reference_forward.py` | numpy self-test of the bit conversion and the factorised forward pass |
| `scripts/inspect_checkpoint.py` | ranks and sizes from a safetensors header, no weights needed |
| `docs/FORMAT.md` | the GGUF layout |
| `docs/LAYERWISE.md` | experiment report: a cheaper layer-wise training route on a 6 GB laptop GPU, with its failures |
| `scripts/layerwise_littlebit.py`, `scripts/finetune_scales.py` | the layer-wise trainer and the global scale pass (experimental, not the papers' recipe) |
| `scripts/export_layerwise_checkpoint.py` | trained blocks → a checkpoint in the LittleBit file layout |
| `scripts/run_pilot_baselines.sh` | every command behind the comparison table in `docs/LAYERWISE.md` |
| `results/` | logs behind every number in this repository |

## Results

Checkpoint: [LittleBitLLM/littlebit-qwen3-8b-0.55bpw](https://huggingface.co/LittleBitLLM/littlebit-qwen3-8b-0.55bpw)
(Qwen3-8B at 0.55 bits per weight, residual path, 0.5 epoch of distillation; CC BY-NC 4.0).
Hardware: laptop, Intel Core i7-13700H (6 threads used), NVIDIA RTX A1000 6 GB, Linux. Builds: `-DGGML_NATIVE=ON`, Release; the GPU build adds `-DGGML_CUDA=ON` and runs with `-ngl 99`. Text: WikiText-2 test,
windows of 2048 tokens, second half of each window scored (`llama-perplexity` protocol).

**The conversion is faithful.** Running perplexity after each of the first four windows:

| Window | llama.cpp, converted GGUF (CPU) | PyTorch float32 reference, original checkpoint |
|---|---|---|
| 1 | 23.5750 | 23.6013 |
| 2 | 25.8926 | 25.8977 |
| 3 | 27.9020 | 27.8870 |
| 4 | 26.2801 | 26.2874 |

**Size, quality and speed** (speed from `llama-bench`, prompt 512 tokens; full test set = 145 windows, on CUDA):

| Model | File | PPL, 12 windows | PPL, full test set | CPU prompt / gen (t/s) | GPU prompt / gen (t/s) |
|---|---|---|---|---|---|
| LittleBit 0.55 bpw, F16 embeddings | 3086 MB | 20.86 (CPU), 20.87 (GPU) | 22.26 ± 0.18 | not rerun | 1152 / 46.7 |
| LittleBit 0.55 bpw, Q8_0 embeddings | 1919 MB | 20.88 | not measured | not measured | 1130 / 52.9 |
| LittleBit 0.55 bpw, Q4_K_M embeddings | 1457 MB | 20.94 | 22.31 ± 0.18 | 34.8 / 18.1 | 1129 / 57.8 |
| Qwen3-8B Q4_K_M (standard, for scale) | 5028 MB | 8.50 | 8.78 ± 0.07 | 40.8 / 7.7 | 1155 / 30.0 |

Reading the table:

- The sub-1-bit model generates about twice as fast as the standard 4-bit one and is 3.5 times smaller.
  Prompt processing is not faster: equal on GPU, about 15% slower on CPU.
- Its quality is far worse: full-test-set perplexity 22.3 against 8.78. That is the checkpoint, not the conversion.
  Its model card reports ARC-Challenge below chance. It is a demonstration of the format, not a usable model.
- The compressed layers are 579 MB. The embedding and output head are not compressed by LittleBit and
  dominate the file; quantizing them to Q4_K_M costs 0.05 perplexity on the full test set.

**At the same file size a standard smaller model is better.** This is the comparison that matters for anyone
choosing a file, and it goes against the sub-1-bit checkpoint. Full test sets, CUDA, same protocol; the second text
is C4 validation (138 windows), which neither model was tuned on here. Logs: `results/same-size/`.

| Model | File | WikiText-2 PPL | C4 PPL | GPU prompt / gen (t/s) |
|---|---|---|---|---|
| LittleBit Qwen3-8B 0.55 bpw, Q2_K embeddings | 1311 MB | 22.41 ± 0.18 | 33.15 ± 0.28 | 1131 / 57.7 |
| Standard Qwen3-1.7B Q4_K_M | 1282 MB | 15.02 ± 0.14 | 21.08 ± 0.19 | 4783 / 119.0 |
| LittleBit Qwen3-8B 0.55 bpw, Q4_K_M embeddings | 1457 MB | 22.31 ± 0.18 | 33.06 ± 0.28 | 1129 / 57.8 |
| Standard Qwen3-8B Q4_K_M | 5028 MB | 8.78 ± 0.07 | 14.33 ± 0.12 | 1155 / 30.0 |

- The 1.7B model at 4 bits has lower perplexity on both texts and generates twice as fast as the 8B model at
  0.55 bits in the same number of megabytes. "Twice as fast and 3.5 times smaller" above is true only against
  the same 8B model at 4 bits.
- [ScarionWorkspace](https://github.com/ScarionWorkspace/littlebit-gguf/blob/main/results/baselines/TABLE.md)
  measured the same pair on other hardware with a different scoring rule (all positions) and found the same
  ordering (24.70 against 16.90). The ratio between the two models is 1.46 there and 1.49 here.
- This checkpoint was trained for half an epoch. A longer-trained or larger checkpoint may change the picture;
  none is published that we could test.

Not measured: other backends (Metal, Vulkan), other architectures, other checkpoints.

## Cross-checks with GeniusDevam/littlebit-gguf

Requested in [their issue #1](https://github.com/GeniusDevam/littlebit-gguf/issues/1). Logs: `results/geniusdevam/`.

| Check | Result |
|---|---|
| Our 8B GGUF files (F16 and Q4_K_M embeddings) in their build at `8853f81` | Load and generate |
| Perplexity of our F16 file, 12 windows, CPU | 20.8561 in their build and in ours; all 12 windows equal to four decimals |
| Same on CUDA (their tree built with `-DGGML_CUDA=ON`) | 20.8654 in both builds |
| Speed on CUDA, same session, pp512 / tg64 | theirs 1133 / 47.0 and ours 1152 / 47.0 (F16 file); theirs 1145 / 57.1 and ours 1173 / 58.8 (Q4_K_M embeddings) |
| Their converter at `8853f81` on the original checkpoint | Stops: `model.layers.1.mlp.up_proj: nonzero scale rounds to zero in fp16` |
| Their converter at `893866f` on the original checkpoint | Converts; reports 56 scales flushed to fp16 zero |
| Their `893866f` file against ours, tensor by tensor | Same size to the byte, same 1659 tensors and metadata keys; 1299 tensors byte-identical; 360 differ only in padding sign bits (columns 208–223 and 824–831 of `lb_u`), which multiply zeros |
| Their `893866f` file in our build, 12 windows, CUDA | 20.8654, same as our file |

The two converters and the two loaders are interchangeable for this checkpoint.

## Cross-checks with ScarionWorkspace's fork

Requested in the same issue thread: their fused GPU kernels had only run on AMD hardware. Their fork
(`ScarionWorkspace/llama.cpp`, branch `littlebit`, `9699b11f5`) built with `-DGGML_CUDA=ON` on the RTX A1000 6 GB.
"Unfused" means `GGML_CUDA_DISABLE_FUSION=1`. Logs: `results/scarion/`.

| Check | Result |
|---|---|
| Builds with CUDA; their converter on the original checkpoint | Yes; 2917 MiB (F16), 1364 MiB with Q4_K_M embeddings |
| Their backend tests, `test-backend-ops -o LITTLEBIT_LINEAR -b CUDA0` | 56 of 56 pass |
| Perplexity, usual settings, 12 windows | 20.8654, same as our build (the fused path is not used at this batch size) |
| Perplexity in batches of 8 tokens, 4 windows, fused | 23.6015, 25.8979, 27.8874, 26.2876: within 0.0004 of the PyTorch float32 reference above |
| Same, unfused | 23.6236, 25.9089, 27.9168, 26.2924 |
| Generation speed (tg64), their F16 file | **24.9 t/s fused, 48.2 t/s unfused** |
| Generation speed (tg64), their Q4_K_M-embedding file | **27.9 t/s fused, 61.5 t/s unfused** |
| Our own files in their build | Load and run; same speeds within noise (24.6 / 49.0 and 27.7 / 62.3) |
| Their per-layer micro-benchmark, q/o projection, 1 token | 99.7 µs fused, 37.9 µs unfused, 63.4 µs dense Q4_K |

On this NVIDIA card the fused path is the more exact one, as they report, but it is about half the speed of the
unfused path, the opposite of their result on an AMD RX 9070 XT (82 to 140 t/s). Ranks that are not multiples of
32 (208 and 824 in this checkpoint) caused no problem.

## Layer-wise training experiment

Separately from the plumbing, we tried compressing a model ourselves on the 6 GB laptop GPU with a cheaper
method than the papers' end-to-end training. Full report with every run, including the failed ones:
[docs/LAYERWISE.md](docs/LAYERWISE.md). In short, for Qwen3-0.6B at 1.08 bits per weight in the compressed layers
(7 hours of training):

- At 200 MB it scores 52.5 on WikiText-2 where the standard quantizations of similar size are broken
  (`IQ2_XXS`, 229 MB: 799).
- Standard `Q2_K` at 296 MB is clearly better (34.4), and on text unlike its calibration data our model degrades
  badly: 194.5 on C4 against 49.1 for `Q2_K` and 27.3 for the original.
- It has not been tried on a larger model, which is where sub-1-bit methods are supposed to pay off.

## How it works

A LittleBit layer is `y = u1 * (U (mid * (V (v2 * x))))` with sign matrices `U`, `V` and `mid = v1 * u2`,
plus the same again for a residual path. `Q1_0` stores 128 sign bits and one fp16 scale per block, so the
row scales fold into the block scale: `V' = diag(mid) V`, `U' = diag(u1) U`. A layer becomes one
element-wise multiply and two ordinary `Q1_0` matrix multiplies per path. The scale-folding idea is from
the design notes in [GeniusDevam/littlebit-gguf](https://github.com/GeniusDevam/littlebit-gguf).

The patch hooks two places: `llama_model_base::create_tensor` loads `<base>.lb_u/lb_v/lb_g` when
`<base>.weight` is absent, and `llm_graph_context::build_lora_mm` builds the factorised graph for those
weights. Architecture code is untouched, so any architecture whose linears go through `build_lora_mm`
should work; only Qwen3 is tested.

## Reproduce

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp && git checkout 9183a55
git apply ../patches/littlebit-q1_0.diff
cmake -B build -DGGML_CUDA=ON && cmake --build build -j --target llama-completion llama-perplexity llama-bench llama-quantize
cd ..

pip install numpy torch transformers safetensors sentencepiece pyyaml tqdm requests
hf download LittleBitLLM/littlebit-qwen3-8b-0.55bpw --local-dir checkpoint

python scripts/convert_littlebit_to_gguf.py checkpoint littlebit-f16.gguf --llama-cpp llama.cpp
python scripts/convert_littlebit_to_gguf.py checkpoint littlebit-q4.gguf --llama-cpp llama.cpp \
    --embd-quant Q4_K_M --quantize-bin llama.cpp/build/bin/llama-quantize

# wiki.test.raw: the `text` column of the WikiText-2 raw test split
# (huggingface.co/datasets/Salesforce/wikitext, config wikitext-2-raw-v1), rows concatenated without separators
llama.cpp/build/bin/llama-perplexity -m littlebit-f16.gguf -f wiki.test.raw -c 2048 -b 2048 --chunks 4
python scripts/reference_ppl.py checkpoint wiki.test.raw --ctx 2048 --chunks 4
```

## Known limits

- 3224 scale values in this checkpoint are below the fp16 normal range (smallest 4.7e-11) and lose
  precision in the `Q1_0` block scale; 56 of them round to zero. No effect is visible in perplexity.
- Ranks are padded to a multiple of 128 (544→640, 208→256, 824→896): 579 MB instead of 464 MB.
- No validation in the loader: a malformed file is not rejected with a clear error.
- LoRA adapters on factorised weights are ignored. Biases on factorised layers are untested.
- Layers must have an input width that is a multiple of 128.
- CPU speed for the F16 and Q8_0 variants, and memory use, were not measured.
- The 12-window numbers come from different backends (LittleBit Q8_0 and Q4_K_M on CPU, the baseline on CUDA); the full-test-set column is all CUDA.

## Licence, provenance and credits

**Licence.** Everything in this repository is MIT: use it, fork it, change it, ship it. See [LICENSE](LICENSE).

**What is not covered.** No model weights are included. A checkpoint you download keeps its own licence, and a
GGUF you convert from it is a derivative of that checkpoint under the same terms. The checkpoint tested here is
CC BY-NC 4.0 (non-commercial). Which weights you run is your decision and your responsibility, not this code's.

**Provenance of the code.**

- All scripts in `scripts/` were written for this repository. No code was copied from
  [SamsungLabs/LittleBit](https://github.com/SamsungLabs/LittleBit) or its forks, which are CC BY-NC 4.0.
  A line-by-line comparison of this repository against SamsungLabs/LittleBit, kimwin2/lb_kernels,
  kimwin2/LittleSpec and GeniusDevam/littlebit-gguf finds no shared code lines beyond one standard
  PyTorch call (`torch.set_num_threads(args.threads)`).
- `patches/littlebit-q1_0.diff` is a change to [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT) and uses
  its existing structures and style.
- The code was written with AI assistance (Claude Code).

**Ideas and facts borrowed, with thanks.**

- The method: LittleBit ([arXiv:2506.13771](https://arxiv.org/abs/2506.13771), NeurIPS 2025) and LittleBit-2
  ([arXiv:2603.00042](https://arxiv.org/abs/2603.00042), ICML 2026) by Lee, Kim, You and Kim (Samsung Research).
- The checkpoint file layout, bit packing and forward-pass formula: learned by reading the official
  SamsungLabs/LittleBit code. Facts about a format, not code.
- Folding the row scales into the `Q1_0` block scale, the tensor names (`lb_g`, `lb_v`, `lb_u`, `_r`) and the
  `littlebit.*` metadata keys: from the design notes in
  [GeniusDevam/littlebit-gguf](https://github.com/GeniusDevam/littlebit-gguf) (MIT), adopted so files can stay compatible. Cross-loading has been run in both directions for the 8B checkpoint (see [Cross-checks](#cross-checks-with-geniusdevamlittlebit-gguf)).
- The `Q1_0` tensor type and its CPU and CUDA kernels: llama.cpp, added for PrismML's 1-bit Bonsai models.
- The test checkpoint: [LittleBitLLM/littlebit-qwen3-8b-0.55bpw](https://huggingface.co/LittleBitLLM/littlebit-qwen3-8b-0.55bpw).
- The baseline: [Qwen/Qwen3-8B-GGUF](https://huggingface.co/Qwen/Qwen3-8B-GGUF).

## Citation

Please cite the original papers:

```bibtex
@inproceedings{lee2025littlebit,
  title     = {LittleBit: Ultra Low-Bit Quantization via Latent Factorization},
  author    = {Lee, Banseok and Kim, Dongkyu and You, Youngcheon and Kim, Youngmin},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2025}
}

@inproceedings{lee2026littlebit2,
  title     = {LittleBit-2: Maximizing the Spectral Energy Gain in Sub-1-Bit LLMs via Latent Geometry Alignment},
  author    = {Lee, Banseok and Kim, Youngmin},
  booktitle = {International Conference on Machine Learning},
  year      = {2026}
}
```
