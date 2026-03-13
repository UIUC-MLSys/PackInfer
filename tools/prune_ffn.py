import argparse
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import init_empty_weights


@torch.no_grad()
def materialize_model(model: torch.nn.Module):
    for name, p in list(model.named_parameters(recurse=True)):
        if getattr(p, "device", None) and p.device.type == "meta":
            p_tensor = torch.zeros(p.shape, dtype=torch.float16, device="cpu")
            parent = model
            *prefix, last = name.split(".")
            for a in prefix:
                parent = getattr(parent, a)
            setattr(parent, last, torch.nn.Parameter(p_tensor, requires_grad=False))
    for name, b in list(model.named_buffers(recurse=True)):
        if getattr(b, "device", None) and b.device.type == "meta":
            b_tensor = torch.zeros(b.shape, dtype=torch.float16, device="cpu")
            parent = model
            *prefix, last = name.split(".")
            for a in prefix:
                parent = getattr(parent, a)
            if hasattr(parent, last):
                delattr(parent, last)
            parent.register_buffer(last, b_tensor, persistent=True)
    if hasattr(model, "tie_weights"):
        try:
            model.tie_weights()  # ty:ignore[call-non-callable]
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--target_intermediate_size", type=int, required=True)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    config = AutoConfig.from_pretrained(
        args.model_id, trust_remote_code=True, local_files_only=False
    )
    if getattr(config, "intermediate_size", None) is None:
        raise ValueError("config.intermediate_size Not Found")
    config.intermediate_size = int(args.target_intermediate_size)

    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

    materialize_model(model)
    model.save_pretrained(out.as_posix(), safe_serialization=True)

    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    tok.save_pretrained(args.output_dir)

    print(f"Saved to: {out.resolve()}")
    print(f"New intermediate_size: {config.intermediate_size}")


if __name__ == "__main__":
    """
    python3 tools/prune_ffn.py \
        --model_id ~/huggingface/Qwen3-4B/ \
        --target_intermediate_size 1024 \
        --output_dir ~/huggingface/Qwen3-4B-fp16-pruned/
    """
    main()
