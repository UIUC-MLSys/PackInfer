import argparse
import json
import os
from pathlib import Path
from typing import List

import numpy as np

from packedinfer.nanovllm import LLM, SamplingParams
from packedinfer.nanovllm.config import Config
from packedinfer.nanovllm.engine.model_runner import _EVENTS, _RUN_STATS
from packedinfer.nanovllm.layers.attention import _ATTN_EVENTS


def load_prompts(path: str, limit: int, truncate_ratio: float) -> List[str]:
    """Load prompts from json/jsonl/text; fallback to 'prompt' then 'instruction'."""
    p = Path(path)
    prompts: List[str] = []
    if p.suffix == ".jsonl":
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                prompt = obj.get("prompt") or obj.get("instruction")
                if prompt:
                    prompts.append(prompt)
    elif p.suffix == ".json":
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data:
                prompt = item.get("prompt") or item.get("context") or item.get("instruction")
                if prompt:
                    prompts.append(prompt)
    else:  # plain text, one prompt per line
        with p.open("r", encoding="utf-8") as f:
            prompts = [line.rstrip("\n") for line in f if line.strip()]

    if truncate_ratio > 0 and truncate_ratio < 1:
        prompts = [p[: max(1, int(len(p) * truncate_ratio))] for p in prompts]
    return prompts[:limit]


def run_backend(
    backend: str,
    model_path: str,
    prompts: List[str],
    max_new_tokens: int,
    temperature: float,
    tp_size: int,
    enable_bfd: bool,
    gpu_mem_util: float,
    max_num_seqs: int,
    packed_tok_threshold: int,
    use_prepacking: bool = False,
):
    os.environ["ATTN_BACKEND"] = backend
    if enable_bfd:
        os.environ["ENABLE_BFD"] = "1"
    else:
        os.environ.pop("ENABLE_BFD", None)

    os.environ["PROFILE_TIMING"] = "1"

    sampling_params = SamplingParams(temperature=temperature, max_tokens=max_new_tokens, ignore_eos=True)
    llm = LLM(model_path, enforce_eager=True, tensor_parallel_size=tp_size, gpu_memory_utilization=gpu_mem_util, use_prepacking=use_prepacking)

    # Warmup
    llm.generate([[42] * 42], sampling_params)

    _ATTN_EVENTS.clear()
    _EVENTS.clear()
    _RUN_STATS.clear()

    cfg = Config(model=model_path, max_num_seqs=max_num_seqs, packed_tok_threshold=packed_tok_threshold, use_prepacking=use_prepacking)
    num_layers = cfg.hf_config.num_hidden_layers

    # Benchmark
    llm.generate(prompts, sampling_params)

    assert len(_ATTN_EVENTS) % num_layers == 0
    attn_events = [_ATTN_EVENTS[i : i + num_layers] for i in range(0, len(_ATTN_EVENTS), num_layers)]

    metrics = []

    prefill_time = 0.0
    decode_tot_time = 0.0
    decode_tot_attn_time = 0.0
    decode_token_count = 0
    for run_stat, event, attn_event in zip(_RUN_STATS, _EVENTS, attn_events):
        if run_stat["is_prefill"]:
            if prefill_time > 0.0:
                metrics.append(
                    {
                        "batch_size": run_stat["batch_size"],
                        "token_num": decode_token_count,
                        "ttft": prefill_time,
                        "tbt": decode_tot_time / decode_token_count,
                        "ttlt": prefill_time + decode_tot_time,
                        "tpt": decode_token_count * run_stat["batch_size"] / decode_tot_time * 1000,
                    }
                )
            prefill_time = event[0].elapsed_time(event[1])
            decode_tot_attn_time = 0
            decode_tot_time = 0
            decode_token_count = 1
        else:
            decode_time = event[0].elapsed_time(event[1])
            decode_tot_time += decode_time
            decode_attn_time = sum(e["start"].elapsed_time(e["end"]) for e in attn_event)
            decode_tot_attn_time += decode_attn_time
            decode_token_count += 1

    # Last batch
    metrics.append(
        {
            "batch_size": run_stat["batch_size"],
            "token_num": decode_token_count,
            "ttft": prefill_time,
            "tbt": decode_tot_time / decode_token_count,
            "ttlt": prefill_time + decode_tot_time,
            "tpt": decode_token_count * run_stat["batch_size"] / decode_tot_time * 1000,
        }
    )

    ttfts = [m["ttft"] for m in metrics]
    tbts = [m["tbt"] for m in metrics]
    ttlts = [m["ttlt"] for m in metrics]
    tpts = [m["tpt"] for m in metrics]

    return {
        "backend": backend,
        "num_prompts": len(prompts),
        "avg_ttft": float(np.mean(ttfts)),
        "p5_ttft": float(np.percentile(ttfts, 5)),
        "p10_ttft": float(np.percentile(ttfts, 10)),
        "p25_ttft": float(np.percentile(ttfts, 25)),
        "p50_ttft": float(np.percentile(ttfts, 50)),
        "p75_ttft": float(np.percentile(ttfts, 75)),
        "p99_ttft": float(np.percentile(ttfts, 99)),
        "p95_ttft": float(np.percentile(ttfts, 95)),
        "p90_ttft": float(np.percentile(ttfts, 90)),
        "avg_tbt": float(np.mean(tbts)),
        "p5_tbt": float(np.percentile(tbts, 5)),
        "p10_tbt": float(np.percentile(tbts, 10)),
        "p25_tbt": float(np.percentile(tbts, 25)),
        "p50_tbt": float(np.percentile(tbts, 50)),
        "p75_tbt": float(np.percentile(tbts, 75)),
        "p99_tbt": float(np.percentile(tbts, 99)),
        "p95_tbt": float(np.percentile(tbts, 95)),
        "p90_tbt": float(np.percentile(tbts, 90)),
        "avg_ttlt": float(np.mean(ttlts)),
        "p5_ttlt": float(np.percentile(ttlts, 5)),
        "p10_ttlt": float(np.percentile(ttlts, 10)),
        "p25_ttlt": float(np.percentile(ttlts, 25)),
        "p50_ttlt": float(np.percentile(ttlts, 50)),
        "p75_ttlt": float(np.percentile(ttlts, 75)),
        "p99_ttlt": float(np.percentile(ttlts, 99)),
        "p95_ttlt": float(np.percentile(ttlts, 95)),
        "p90_ttlt": float(np.percentile(ttlts, 90)),
        "avg_tpt": float(np.mean(tpts)),
        "p5_tpt": float(np.percentile(tpts, 5)),
        "p10_tpt": float(np.percentile(tpts, 10)),
        "p25_tpt": float(np.percentile(tpts, 25)),
        "p50_tpt": float(np.percentile(tpts, 50)),
        "p75_tpt": float(np.percentile(tpts, 75)),
        "p99_tpt": float(np.percentile(tpts, 99)),
        "p95_tpt": float(np.percentile(tpts, 95)),
        "p90_tpt": float(np.percentile(tpts, 90)),
    }


def main():
    parser = argparse.ArgumentParser(description="Run PackInfer TTFT/TBT/TTLT and throughput benchmarks")
    parser.add_argument("--model", default="~/huggingface/Qwen3-30B-A3B-fp16-pruned/", help="Model path")
    parser.add_argument("--prompts", default="data/dataset/lmsys.json", help="Prompt source")
    parser.add_argument("--num-prompts", type=int, default=10000, help="Number of prompts to run")
    parser.add_argument("--truncate-ratio", type=float, default=0.5, help="Portion of each prompt to keep (0-1]")
    parser.add_argument("--max-new-tokens", type=int, default=32, help="Max new tokens per request")
    parser.add_argument("--temperature", type=float, default=1e-9, help="Sampling temperature")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--backend", type=str, default="FLASH_ATTN", help="Backends to test")
    parser.add_argument("--enable-bfd", action="store_true", help="Enable best-fit decreasing packing")
    parser.add_argument("--out-dir", default="data", help="Where to save JSON results")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8, help="Fraction of GPU memory budget for KV cache")
    parser.add_argument("--max-num-seqs", type=int, default=256, help="Max number of sequences in a batch")
    parser.add_argument("--packed-tok-threshold", type=int, default=8192, help="Packed attention token threshold")
    parser.add_argument("--use-prepacking", action="store_true", help="Enable prepacking baseline")
    args = parser.parse_args()

    prompts = load_prompts(args.prompts, args.num_prompts, args.truncate_ratio)
    if not prompts:
        raise RuntimeError("No prompts loaded; check --prompts path")

    result = run_backend(
        backend=args.backend,
        model_path=os.path.expanduser(args.model),
        prompts=prompts,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        tp_size=args.tensor_parallel_size,
        enable_bfd=args.enable_bfd,
        gpu_mem_util=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        packed_tok_threshold=args.packed_tok_threshold,
        use_prepacking=args.use_prepacking,
    )

    print(f"Backend: {args.backend}")
    print(f"TTFT ms: {result['avg_ttft']}")
    print(f"TBT  ms: {result['avg_tbt']}")
    print(f"TTLT ms: {result['avg_ttlt']}")

    model_dir_name = Path(os.path.expanduser(args.model)).name
    out_dir = Path(args.out_dir) / model_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.use_prepacking:
        out_path = out_dir / f"{Path(args.prompts).stem}_prepacking.json"
    else:
        out_path = out_dir / f"{Path(args.prompts).stem}_{args.backend.lower()}.json"
    out_path.write_text(json.dumps(result, indent=4))
    print(f"Saved results to {out_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
