#!/bin/bash

set -euo pipefail

OUT_BASE="data/ablation"

MODELS=("~/huggingface/Qwen3-4B-fp16-pruned/")
PROMPTS_LIST=("data/dataset/alpaca.json")
BACKENDS=("PACKED_ATTN")

GROUP_SIZES=(8192 4096 2048 1024 512)

for model in "${MODELS[@]}"; do
	for prompts in "${PROMPTS_LIST[@]}"; do
		for backend in "${BACKENDS[@]}"; do
			for gs in "${GROUP_SIZES[@]}"; do
				python benchmark/metrics.py \
					--backend "${backend}" \
					--prompts "${prompts}" \
					--model "${model}" \
					--packed-tok-threshold "${gs}" \
					--out-dir "${OUT_BASE}/group_size_${gs}"
			done
		done
	done
done