#!/usr/bin/env bash
# Install the isolated, user-scoped vLLM runtime used by the local Windows
# launcher.  No sudo or system Python changes are required.
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
runtime_root="${VLLM_RUNTIME_ROOT:-$HOME/vllm-runtime}"
venv_root="$runtime_root/.venv"
mkdir -p "$runtime_root/logs" "$runtime_root/huggingface"

# uv creates an isolated environment and handles dependency resolution without
# relying on the distro pip, which is broken in this WSL image.
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
uv venv --python /usr/bin/python3 "$venv_root"
uv pip install --python "$venv_root/bin/python" --upgrade vllm
"$venv_root/bin/python" - <<'PY'
import torch
import vllm

print(f"vLLM {vllm.__version__}")
print(f"PyTorch {torch.__version__}; CUDA available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in WSL; vLLM cannot serve the local VLM")
PY
