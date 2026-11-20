# littlebit-llamacpp

A small loader patch and a converter that let llama.cpp run LittleBit / LittleBit-2 sub-1-bit checkpoints,
plus measurements on one published checkpoint.

**Whose work this is.** The compression method is LittleBit and LittleBit-2 by Banseok Lee, Dongkyu Kim,
Youngcheon You and Youngmin Kim at Samsung Research. They developed the method, the training and the
checkpoint format. This repository contributes only plumbing and tests: 64 lines that load their factorised
layers into llama.cpp's existing 1-bit `Q1_0` type, a script that converts their checkpoint files, and
measurements. No new tensor type and no new kernel. Please cite their papers (see [Citation](#citation)).

Research prototype. Not affiliated with or endorsed by Samsung or the llama.cpp project.

**Related project:** [GeniusDevam/littlebit-gguf](https://github.com/GeniusDevam/littlebit-gguf) implements the same
approach independently, with validation, tests and CI for Llama, Qwen2 and Qwen3. The two overlap
(see its issue #1); this repository adds CUDA verification, an independent reference and speed measurements.
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
| `results/` | perplexity logs behind the numbers below |

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

Not measured: other backends (Metal, Vulkan), other architectures, other checkpoints.

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
  precision in the `Q1_0` block scale. No effect is visible in perplexity.
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
  [GeniusDevam/littlebit-gguf](https://github.com/GeniusDevam/littlebit-gguf) (MIT), adopted so files can stay compatible. Cross-loading between the two projects has not been run; by inspection of their patch, a file from this converter meets their loader's checks.
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
