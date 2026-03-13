#!/bin/bash

# Model Download
hf download Qwen/Qwen3-32B --local-dir ~/huggingface/Qwen3-32B-FP8
python3 tools/prune_ffn.py \
        --model_id ~/huggingface/Qwen3-32B-FP8/ \
        --target_intermediate_size 1024 \
        --output_dir ~/huggingface/Qwen3-32B-fp8-pruned/

hf download mistralai/Mistral-7B-Instruct-v0.3 --local-dir ~/huggingface/Mistral-7B/
python3 tools/prune_ffn.py \
        --model_id ~/huggingface/Mistral-7B/ \
        --target_intermediate_size 1024 \
        --output_dir ~/huggingface/Mistral-7B-fp16-pruned/
jq -i '. + {"head_dim": 128}' ~/huggingface/Mistral-7B-fp16-pruned/config.json

set -euo pipefail

source .venv/bin/activate

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BENCH_SCRIPT="$ROOT_DIR/benchmark/metrics.py"

PROMPTS=(
	"$ROOT_DIR/data/dataset/lmsys.json"
	"$ROOT_DIR/data/dataset/alpaca.json"
	"$ROOT_DIR/data/dataset/text2sql.json"
)

BACKENDS=(
	"FLASH_ATTN"
	"PACKED_ATTN"
)

MODELS=(
	~/huggingface/Qwen3-4B-fp16-pruned/
	~/huggingface/Mistral-7B-fp16-pruned/
)

run_bench() {
	local backend="$1"
	local prompts="$2"
	local model="$3"
	python "$BENCH_SCRIPT" --backend "$backend" --prompts "$prompts" --model "$model"
}

for model in "${MODELS[@]}"; do
	for prompts in "${PROMPTS[@]}"; do
		for backend in "${BACKENDS[@]}"; do
			run_bench "$backend" "$prompts" "$model"
		done
	done
done
