#!/bin/bash

set -euo pipefail

OUT_BASE="data/ablation"

MODELS=("~/huggingface/Qwen3-4B-fp16-pruned/")
PROMPTS_LIST=("data/dataset/alpaca.json")
BACKENDS=("PACKED_ATTN")

BATCH_SIZES=(2048 1024 512 256 128)

for model in "${MODELS[@]}"; do
	for prompts in "${PROMPTS_LIST[@]}"; do
		for backend in "${BACKENDS[@]}"; do
			for bs in "${BATCH_SIZES[@]}"; do
				python benchmark/metrics.py \
					--backend "${backend}" \
					--prompts "${prompts}" \
					--model "${model}" \
					--max-num-seqs "${bs}" \
					--out-dir "${OUT_BASE}/batch_size_${bs}"
			done
		done
	done
done
