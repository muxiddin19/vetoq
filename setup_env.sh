#!/usr/bin/env bash
# Recreate the Python environment used for the T1 experiments (Python 3.14 via uv).
# torch is the CUDA 12.6 build (supports older GPUs such as TITAN Xp / sm_61); the paper's numbers are CPU runs.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
uv venv .venv --python 3.14
VIRTUAL_ENV=.venv uv pip install "torch==2.10.0" --index-url https://download.pytorch.org/whl/cu126
VIRTUAL_ENV=.venv uv pip install -r requirements.txt --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu126
.venv/bin/python -c "import pandas, numpy, sklearn, sentence_transformers, splink, torch; print('ok', pandas.__version__, numpy.__version__, sklearn.__version__, sentence_transformers.__version__, splink.__version__, torch.__version__)"
