import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@torch.no_grad()
def convert_to_fp16(src_dir, dst_dir):
    model = AutoModelForCausalLM.from_pretrained(src_dir, dtype=torch.float16, device_map="auto", trust_remote_code=True)
    model.config.torch_dtype = torch.float16
    model.save_pretrained(dst_dir, safe_serialization=True)

    tok = AutoTokenizer.from_pretrained(src_dir, trust_remote_code=True)
    tok.save_pretrained(dst_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    convert_to_fp16(args.model_id, out.as_posix())

    print(f"Saved to: {out.resolve()}")


if __name__ == "__main__":
    """
    python3 tools/convert_to_fp16.py \
        --model_id ~/huggingface/Qwen3-0.6B/ \
        --output_dir ~/huggingface/Qwen3-0.6B-fp16/
    """
    main()
