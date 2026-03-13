# PackInfer

## Installation

```bash
git clone --recurse-submodules https://github.com/UIUC-MLSys/PackInfer.git
uv venv --python 3.13
source .venv/bin/activate
uv pip install torch==2.8 transformers xxhash huggingface_hub accelerate protobuf sentencepiece
uv pip install -Uve . --no-build-isolation
uv pip install flash-attn --no-build-isolation
```

## Evaluation

```bash
scripts/benchmark.sh
```
Results are located in `data/<MODEL_ID>`.
