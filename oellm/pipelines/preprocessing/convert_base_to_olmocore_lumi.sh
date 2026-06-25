#!/usr/bin/env bash
#SBATCH --job-name=hf2olmocore
#SBATCH --account=project_465002530
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=/pfs/lustrep1/scratch/project_462001516/abhasjha/fabio-open-instruct/open-instruct/oellm/pipelines/preprocessing/logs/hf2olmocore_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=abhash.kumar.jha@tue.ellis.eu

# Convert the HF allenai/Olmo-3-7B-Instruct-SFT checkpoint into OLMo-core format
# so OLMo-sft.py can load it as the pretraining checkpoint for Track G SFT.
# One-shot; produces the shared base checkpoint for all five G* runs.
# LUMI-G port of convert_base_to_olmocore_leonardo.sh -- runs inside the ROCm
# PyTorch Singularity container on a single MI250X GCD (small-g).

set -euo pipefail

# --- LUMI container + env -----------------------------------------------------
SCRATCH_ROOT=/pfs/lustrep1/scratch/project_462001516
LUMI_DIR="$SCRATCH_ROOT/abhasjha/lumi-container"
SIF="$LUMI_DIR/lumi-pytorch-rocm-6.2.1-python-3.12-pytorch-20240918-vllm-4075b35.sif"
# Dedicated conversion venv: torch 2.7.1+rocm6.2.4 + transformers 4.57 (needed
# for Olmo3Config) + matching torchvision 0.22.1 + olmo_core deps. Kept separate
# from the training venv (container torch 2.6 / transformers 4.45) so the newer
# stack required only by the HF->OLMo-core conversion can't disturb training.
VENV="$LUMI_DIR/venv-convert"
PROJECT_ROOT="$SCRATCH_ROOT/abhasjha/fabio-open-instruct/open-instruct"
OLMOCORE_PATH="$SCRATCH_ROOT/abhasjha/fabio-open-instruct/OLMo-core"
HF_REPO="allenai/Olmo-3-7B-Instruct-SFT"
OUTPUT_DIR="$PROJECT_ROOT/checkpoints/base/Olmo-3-7B-Instruct-SFT-olmocore"

# HF offline cache (shared) -- compute nodes have no internet.
HF_HOME="${HF_HOME:-$SCRATCH_ROOT/cache/huggingface/abhasjha}"
# The container only binds the /pfs path; an inherited /scratch/<proj> form (a
# symlink on the host) is invisible inside it -> normalize to the canonical path.
case "$HF_HOME" in
  /scratch/project_462001516/*) HF_HOME="$SCRATCH_ROOT/${HF_HOME#/scratch/project_462001516/}" ;;
esac
export HF_HOME
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# olmo_core's load_hf_model() calls repo_exists() (a network call) when given a
# repo id, which fails under offline mode. Pass the local snapshot directory
# instead so it takes the local-path branch and skips the network entirely.
HF_SNAPSHOT_DIR=$(ls -d "$HF_HOME"/hub/models--allenai--Olmo-3-7B-Instruct-SFT/snapshots/*/ 2>/dev/null | head -1)
[ -n "$HF_SNAPSHOT_DIR" ] || { echo "ERROR: local snapshot for $HF_REPO not found under $HF_HOME/hub"; exit 1; }

# Validate the conversion in fp32: at bf16 the per-layer matmul noise alone
# exceeds the script's 1e-4 tolerance (a correct conversion still "fails"),
# whereas fp32 cleanly confirms the OLMo-core and HF forward passes agree.
export OLMOCORE_VALIDATE_FP32=1

# Resolve `import olmo_core` to the team fork (has the SFT + conversion code and
# the olmo3_7b arch registration, including the YaRN RoPE scaling on the
# full-attention layers required to match the released checkpoint).
export PYTHONPATH="${OLMOCORE_PATH}/src:${PYTHONPATH:-}"

mkdir -p "$OUTPUT_DIR" "$PROJECT_ROOT/oellm/pipelines/preprocessing/logs"

echo "=============================================="
echo "HF -> OLMo-core conversion (LUMI)"
echo "  HF model : $HF_REPO"
echo "  Snapshot : $HF_SNAPSHOT_DIR"
echo "  Arch     : olmo3_7b   Tokenizer: dolma2"
echo "  Output   : $OUTPUT_DIR"
echo "=============================================="

# Singularity inherits the exported env above (HF_*, PYTHONPATH). On ROCm the
# device string is still 'cuda' (HIP maps it to the MI250X GCD).
singularity exec \
  --bind "$SCRATCH_ROOT:$SCRATCH_ROOT" \
  "$SIF" bash -c "
    set -euo pipefail
    source /opt/miniconda3/bin/activate pytorch
    source '$VENV/bin/activate'
    # ray/olmo_core reject LUMI's ROCR_VISIBLE_DEVICES and demand HIP_VISIBLE_DEVICES.
    if [ -n \"\${ROCR_VISIBLE_DEVICES:-}\" ]; then
        export HIP_VISIBLE_DEVICES=\"\$ROCR_VISIBLE_DEVICES\"
        unset ROCR_VISIBLE_DEVICES
    fi
    cd '$PROJECT_ROOT'
    python '${OLMOCORE_PATH}/src/examples/huggingface/convert_checkpoint_from_hf.py' \
        --checkpoint-input-path '$HF_SNAPSHOT_DIR' \
        --model-arch olmo3_7b \
        --tokenizer dolma2 \
        --output-dir '$OUTPUT_DIR' \
        --device cuda
  "

echo "CONVERSION COMPLETE -> $OUTPUT_DIR"
