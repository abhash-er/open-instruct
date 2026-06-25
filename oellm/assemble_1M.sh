#!/usr/bin/env bash
# Assemble the five synthif 1M-scale mixtures from the by_language source pool.
#
# Same English-ratio sweep as the 500k set (0/25/50/75/100% English), but
# 1,000,000 samples each. CPU-only work; runs inside the LUMI ROCm container's
# layering venv (needs pyarrow + oellm's language_mixer). Writes
# synthif-1M-*.parquet next to the 500k files in $GEN/assembled.
#
# Usage:
#   oellm/assemble_1M.sh              # assemble all five 1M mixtures
#   oellm/assemble_1M.sh --dry-run    # preview counts without writing
set -euo pipefail

SCRATCH=/pfs/lustrep1/scratch/project_462001516
LUMI_DIR="$SCRATCH/abhasjha/lumi-container"
SIF="$LUMI_DIR/lumi-pytorch-rocm-6.2.1-python-3.12-pytorch-20240918-vllm-4075b35.sif"
VENV="$LUMI_DIR/venv"
PROJ="$SCRATCH/abhasjha/fabio-open-instruct/open-instruct"
GEN="$SCRATCH/abhasjha/gen_outputs/outputs/open_instruct"

# Forward any extra args (e.g. --dry-run) to the assembler.
EXTRA_ARGS="$*"

singularity exec --bind "$SCRATCH:$SCRATCH" "$SIF" bash -c "
    set -euo pipefail
    source /opt/miniconda3/bin/activate pytorch
    source '$VENV/bin/activate'
    cd '$PROJ'
    # assemble_mixture.py does 'from oellm.utils...'; put the repo root on the path.
    export PYTHONPATH='$PROJ':\${PYTHONPATH:-}
    for cfg in oellm/configs/synthif_1M_*.yaml; do
        echo \"=== Assembling \$cfg ===\"
        python oellm/pipelines/preprocessing/assemble_mixture.py \
            --config \"\$cfg\" \
            --by-language-dir '$GEN/by_language' \
            --output-dir '$GEN/assembled' \
            $EXTRA_ARGS
    done
"
