#!/bin/bash

set -euo pipefail

source .venv/bin/activate

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BENCH_SCRIPT="$ROOT_DIR/benchmark/metrics.py"
OUT_BASE="data/ablation"

PROMPTS=(
	"$ROOT_DIR/data/dataset/lmsys.json"
)

BACKENDS=(
	"FLASH_ATTN"
	"PACKED_ATTN"
)

TP_SIZE=(
	1
    2
    4
)

run_bench() {
	local backend="$1"
	local prompts="$2"
	local tp="$3"
	python "$BENCH_SCRIPT" \
	--backend "$backend" \
	--prompts "$prompts" \
	--model "~/huggingface/Qwen3-32B-fp8-pruned/" \
	--tensor-parallel-size "$tp" \
	--max-num-seqs 256 \
	--out-dir "${OUT_BASE}/tp${tp}-Qwen3-32B-fp8-pruned"
}

for tp in "${TP_SIZE[@]}"; do
	for prompts in "${PROMPTS[@]}"; do
		for backend in "${BACKENDS[@]}"; do
			run_bench "$backend" "$prompts" "$tp"
		done
	done
done
