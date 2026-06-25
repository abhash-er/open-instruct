#!/usr/bin/env bash
#SBATCH --job-name=synthif-olmocore2hf
#SBATCH --account=project_465002530
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=/pfs/lustrep1/scratch/project_462001516/abhasjha/fabio-open-instruct/open-instruct/oellm/experiments/dolci_synthetic/logs/olmocore2hf_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=abhash.kumar.jha@tue.ellis.eu

# Convert a trained Track-G (dolci-synthetic) OLMo-core checkpoint back to HF
# format so it can be evaluated / shared. Inverse of the base conversion in
# oellm/pipelines/preprocessing/convert_base_to_olmocore_lumi.sh; runs inside the
# same ROCm container + venv-convert (torch 2.7.1+rocm6.2.4, transformers 4.57).
#
# Point it at either a specific step dir or a run folder (it auto-picks the
# latest stepN). Defaults to the layout written by launch_all_train.sh:
#   checkpoints/synthif/synthif-<SCALE>-<EXP>/stepN
#
# Usage:
#   # by experiment + scale (latest step):
#   EXPERIMENT=G1-100en SCALE=500k sbatch oellm/experiments/dolci_synthetic/scripts/convert_trained_to_hf_lumi.sh
#   # explicit checkpoint dir:
#   CKPT_DIR=/.../synthif-1M-G3-50en/step1234 sbatch ...convert_trained_to_hf_lumi.sh

set -euo pipefail

# --- LUMI container + env -----------------------------------------------------
SCRATCH_ROOT=/pfs/lustrep1/scratch/project_462001516
LUMI_DIR="$SCRATCH_ROOT/abhasjha/lumi-container"
SIF="$LUMI_DIR/lumi-pytorch-rocm-6.2.1-python-3.12-pytorch-20240918-vllm-4075b35.sif"
VENV="$LUMI_DIR/venv-convert"
PROJECT_ROOT="$SCRATCH_ROOT/abhasjha/fabio-open-instruct/open-instruct"
OLMOCORE_PATH="$SCRATCH_ROOT/abhasjha/fabio-open-instruct/OLMo-core"
CONVERT_SCRIPT="$OLMOCORE_PATH/src/examples/huggingface/convert_checkpoint_to_hf.py"
TOKENIZER="allenai/Olmo-3-7B-Instruct-SFT"

CKPT_ROOT="${CKPT_ROOT:-$PROJECT_ROOT/checkpoints/synthif}"

# --- resolve the input checkpoint dir -----------------------------------------
if [ -z "${CKPT_DIR:-}" ]; then
  EXPERIMENT="${EXPERIMENT:?Set CKPT_DIR, or EXPERIMENT (+ SCALE) to auto-resolve}"
  SCALE="${SCALE:-500k}"
  RUN_DIR="$CKPT_ROOT/synthif-${SCALE}-${EXPERIMENT}"
  [ -d "$RUN_DIR" ] || { echo "ERROR: run dir not found: $RUN_DIR"; exit 1; }
  # latest step (numeric sort on the trailing integer)
  CKPT_DIR=$(ls -d "$RUN_DIR"/step* 2>/dev/null | sed 's/.*step//' | sort -n | tail -1 | xargs -I{} echo "$RUN_DIR/step{}")
  [ -n "$CKPT_DIR" ] && [ -d "$CKPT_DIR" ] || { echo "ERROR: no step* checkpoints under $RUN_DIR"; exit 1; }
fi

OUTPUT_DIR="${OUTPUT_DIR:-${CKPT_DIR%/}-hf}"

# HF offline cache (compute nodes have no internet); normalize /scratch->/pfs.
HF_HOME="${HF_HOME:-$SCRATCH_ROOT/cache/huggingface/abhasjha}"
case "$HF_HOME" in
  /scratch/project_462001516/*) HF_HOME="$SCRATCH_ROOT/${HF_HOME#/scratch/project_462001516/}" ;;
esac
export HF_HOME
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Validate in fp32: at bf16 per-layer matmul noise exceeds the script's tolerance
# (a correct conversion still "fails"); fp32 cleanly confirms HF == OLMo-core.
export OLMOCORE_VALIDATE_FP32=1
export PYTHONPATH="${OLMOCORE_PATH}/src:${PYTHONPATH:-}"

mkdir -p "$OUTPUT_DIR" "$PROJECT_ROOT/oellm/experiments/dolci_synthetic/logs"

echo "=============================================="
echo "OLMo-core -> HF conversion (Track G, LUMI)"
echo "  Input  : $CKPT_DIR"
echo "  Output : $OUTPUT_DIR"
echo "  Tokenizer: $TOKENIZER"
echo "=============================================="

singularity exec \
  --bind "$SCRATCH_ROOT:$SCRATCH_ROOT" \
  "$SIF" bash -c "
    set -euo pipefail
    source /opt/miniconda3/bin/activate pytorch
    source '$VENV/bin/activate'
    if [ -n \"\${ROCR_VISIBLE_DEVICES:-}\" ]; then
        export HIP_VISIBLE_DEVICES=\"\$ROCR_VISIBLE_DEVICES\"
        unset ROCR_VISIBLE_DEVICES
    fi
    cd '$PROJECT_ROOT'
    python '$CONVERT_SCRIPT' \
        -i '$CKPT_DIR' \
        -o '$OUTPUT_DIR' \
        --tokenizer '$TOKENIZER' \
        --max-sequence-length 32768 \
        --dtype bfloat16 \
        --device cuda
  "

echo "CONVERSION COMPLETE -> $OUTPUT_DIR"
