#!/usr/bin/env bash
# Same-tool comparison for the layer-wise pilot (docs/LAYERWISE.md): the trained Qwen3-0.6B model against
# standard llama.cpp quantizations of the same model, all scored by llama-perplexity with one protocol.
#
# Run from a work directory that has:
#   pilot/qwen3-0.6b/            original Hugging Face model (Qwen/Qwen3-0.6B)
#   pilot/run3-scales-s3000/     block_NNN.safetensors from layerwise_littlebit.py + finetune_scales.py
#   data/wiki.train.raw, data/wiki.test.raw   WikiText-2 raw, rows concatenated without separators
#   data/c4.val.first1300k.txt   first 619 documents of allenai/c4 en/c4-validation.00000-of-00008.json.gz,
#                                joined with blank lines (out-of-domain check)
# and a patched llama.cpp CUDA build (see README) in $LLAMA.
set -euo pipefail
LLAMA=${LLAMA:-../llama.cpp}
B=$LLAMA/build-cuda/bin
PY=${PY:-python}
S=$(dirname "$0")
O=out/pilot
export LD_LIBRARY_PATH=$B PYTHONDONTWRITEBYTECODE=1
mkdir -p $O/ppl

# 1. Trained blocks -> LittleBit checkpoint -> GGUF (three embedding types).
$PY $S/export_layerwise_checkpoint.py pilot/qwen3-0.6b pilot/run3-scales-s3000 pilot/export-run3-scales
for t in f16 q8_0; do
  $PY $S/convert_littlebit_to_gguf.py pilot/export-run3-scales $O/qwen3-0.6b-layerwise-1.08bpw-$t.gguf \
      --llama-cpp $LLAMA --embd-type $t
done
$PY $S/convert_littlebit_to_gguf.py pilot/export-run3-scales $O/qwen3-0.6b-layerwise-1.08bpw-q4_k_m.gguf \
    --llama-cpp $LLAMA --embd-quant Q4_K_M --quantize-bin $B/llama-quantize

# 2. Baselines from the same original model. The Hugging Face file stores the tied output head as a second
#    copy of the embedding; it is dropped first so file sizes are comparable (llama.cpp ties them itself).
$PY - <<'EOF'
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file
src, dst = Path("pilot/qwen3-0.6b"), Path("pilot/qwen3-0.6b-dedup")
dst.mkdir(exist_ok=True)
d = load_file(src / "model.safetensors")
assert torch.equal(d["lm_head.weight"], d["model.embed_tokens.weight"])
del d["lm_head.weight"]
save_file(d, dst / "model.safetensors", metadata={"format": "pt"})
for f in src.iterdir():
    if f.name != "model.safetensors" and f.is_file() and not (dst / f.name).exists():
        (dst / f.name).symlink_to(f.resolve())
EOF
$PY $LLAMA/convert_hf_to_gguf.py pilot/qwen3-0.6b-dedup --outfile $O/qwen3-0.6b-f16.gguf --outtype f16

# Importance matrix from the same text and the same amount of it that the pilot was calibrated on
# (1024 sequences of 512 tokens of WikiText-2 train), so the baselines get the same in-domain advantage.
$B/llama-imatrix -m $O/qwen3-0.6b-f16.gguf -f data/wiki.train.raw -o $O/qwen3-0.6b.imatrix.gguf \
    -ngl 99 -c 512 --chunks 1024 -t 4
for q in Q8_0 Q4_K_M Q3_K_S Q2_K IQ2_XXS IQ1_M IQ1_S TQ1_0 Q1_0; do
  $B/llama-quantize --imatrix $O/qwen3-0.6b.imatrix.gguf $O/qwen3-0.6b-f16.gguf $O/qwen3-0.6b-$q.gguf $q 6
done

# 3. Perplexity: windows of 2048 tokens, second half scored (llama-perplexity default), whole file.
for m in $O/qwen3-0.6b-*.gguf; do
  case $m in *imatrix*) continue ;; esac
  n=$(basename "$m" .gguf)
  for d in wiki:data/wiki.test.raw c4:data/c4.val.first1300k.txt; do
    $B/llama-perplexity -m "$m" -f "${d#*:}" -c 2048 -b 2048 -ngl 99 -t 4 > "$O/ppl/$n.${d%%:*}.log" 2>&1
    printf '%-44s %-5s %s\n' "$n" "${d%%:*}" "$(grep -oE 'Final estimate: PPL = [0-9.e+]+ \+/- [0-9.]+' "$O/ppl/$n.${d%%:*}.log")"
  done
done
