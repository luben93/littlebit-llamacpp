# littlebit-llamacpp

Run a LittleBit / LittleBit-2 sub-1-bit checkpoint in llama.cpp, using the 1-bit `Q1_0` tensor type
llama.cpp already has. No new tensor type and no new kernel: a converter and a 64-line patch.

Research prototype. Not affiliated with Samsung or the llama.cpp project.

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
Hardware: laptop, 20-thread x86 CPU (6 threads used), NVIDIA RTX A1000 6 GB. Text: WikiText-2 test,
windows of 2048 tokens, second half of each window scored (`llama-perplexity` protocol).

**The conversion is faithful.** Running perplexity after each of the first four windows:

| Window | llama.cpp, converted GGUF (CPU) | PyTorch float32 reference, original checkpoint |
|---|---|---|
| 1 | 23.5750 | 23.6013 |
| 2 | 25.8926 | 25.8977 |
| 3 | 27.9020 | 27.8870 |
| 4 | 26.2801 | 26.2874 |

**Size, quality and speed** (12 windows; speed from `llama-bench`, prompt 512 tokens):

| Model | File | PPL, 12 windows | CPU prompt / gen (t/s) | GPU prompt / gen (t/s) |
|---|---|---|---|---|
| LittleBit 0.55 bpw, F16 embeddings | 3086 MB | 20.86 (CPU), 20.87 (GPU) | not rerun | not measured |
| LittleBit 0.55 bpw, Q8_0 embeddings | 1919 MB | 20.88 | not measured | not measured |
| LittleBit 0.55 bpw, Q4_K_M embeddings | 1457 MB | 20.94 | 34.8 / 18.1 | 1129 / 57.8 |
| Qwen3-8B Q4_K_M (standard, for scale) | 5028 MB | 8.50 | 40.8 / 7.7 | 1155 / 30.0 |

Reading the table:

- The sub-1-bit model generates about twice as fast as the standard 4-bit one and is 3.5 times smaller.
- Its quality is far worse: perplexity 20.9 against 8.5. That is the checkpoint, not the conversion.
  Its model card reports ARC-Challenge below chance. It is a demonstration of the format, not a usable model.
- The compressed layers are 579 MB. The embedding and output head are not compressed by LittleBit and
  dominate the file; quantizing them to Q4_K_M costs 0.09 perplexity.

Not measured: full-test-set perplexity, other backends (Metal, Vulkan), other architectures, other checkpoints.

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

llama.cpp/build/bin/llama-perplexity -m littlebit-f16.gguf -f wiki.test.raw -c 2048 -b 2048 --chunks 4
python scripts/reference_ppl.py checkpoint wiki.test.raw --ctx 2048 --chunks 4
```

## Known limits

- 3224 scale values in this checkpoint are below the fp16 normal range (smallest 4.7e-11) and lose
  precision in the `Q1_0` block scale. No effect is visible in perplexity.
- Ranks are padded to a multiple of 128 (544→640, 208→256, 824→896): 579 MB instead of 464 MB.
- LoRA adapters on factorised weights are ignored. Biases on factorised layers are untested.
- Layers must have an input width that is a multiple of 128.

## Licence and credits

Code in this repository: MIT. The patch is a change to llama.cpp (MIT). Model weights keep their own
licence (the checkpoint above is CC BY-NC 4.0); no weights are distributed here.
The checkpoint format and forward pass were learned by reading
[SamsungLabs/LittleBit](https://github.com/SamsungLabs/LittleBit) (CC BY-NC 4.0); no code from it is included.

Papers: LittleBit ([arXiv:2506.13771](https://arxiv.org/abs/2506.13771), NeurIPS 2025) and
LittleBit-2 ([arXiv:2603.00042](https://arxiv.org/abs/2603.00042), ICML 2026), Lee, Kim, You and Kim.
