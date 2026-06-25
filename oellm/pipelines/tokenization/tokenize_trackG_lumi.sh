#!/usr/bin/env bash
#SBATCH --job-name=tokenize-G
#SBATCH --account=project_465002530
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --output=/pfs/lustrep1/scratch/project_462001516/abhasjha/fabio-open-instruct/open-instruct/oellm/pipelines/tokenization/logs/tokenize_trackG_%A_%a.out
#SBATCH --error=/pfs/lustrep1/scratch/project_462001516/abhasjha/fabio-open-instruct/open-instruct/oellm/pipelines/tokenization/logs/tokenize_trackG_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=abhash.kumar.jha@tue.ellis.eu

# Tokenize Track G 500k assembled mixtures (synthetic-IF EU + Dolci English replay)
# into OLMo-core numpy layout. LUMI-G port of tokenize_trackG_leonardo.sh.
#
# Tokenization itself is CPU-bound, but the job MUST run on a GPU partition
# (`small-g`, 1 GPU) anyway: on a GPU-less node (`small`/`debug`) the open_instruct
# import chain wedges at startup -- the process spins forever in the main thread
# right after deepspeed's "TorchCheckpointEngine Initialized" line and never
# reaches tokenization (observed: torch `_inductor` compile-worker pool spawns and
# hangs with no visible accelerator; DS_ACCELERATOR=cpu and the vllm-neutralization
# are not sufficient). With a single GPU visible the same script proceeds normally.
# It executes inside the LUMI ROCm PyTorch Singularity container with open-instruct's
# deps layered on top (the same venv enter.sh builds). Compute nodes have no
# internet, so the Olmo-3 tokenizer is read from the shared HF cache in offline mode.
#
# Usage (SCALE defaults to 500k):
#   sbatch --array=0-4 oellm/pipelines/tokenization/tokenize_trackG_lumi.sh
#   SCALE=1M sbatch --array=0-4 oellm/pipelines/tokenization/tokenize_trackG_lumi.sh

set -euo pipefail

# --- LUMI container + env -----------------------------------------------------
SCRATCH_ROOT=/pfs/lustrep1/scratch/project_462001516
LUMI_DIR="$SCRATCH_ROOT/abhasjha/lumi-container"
SIF="$LUMI_DIR/lumi-pytorch-rocm-6.2.1-python-3.12-pytorch-20240918-vllm-4075b35.sif"
VENV="$LUMI_DIR/venv"
PROJECT_ROOT="$SCRATCH_ROOT/abhasjha/fabio-open-instruct/open-instruct"
ASSEMBLED_DIR="$SCRATCH_ROOT/abhasjha/gen_outputs/outputs/open_instruct/assembled"
TOKENIZER="allenai/Olmo-3-7B-Instruct-SFT"
MAX_SEQ_LENGTH=32768

# Scale of the mixture to tokenize. Selects the assembled file
# synthif-<SCALE>-<suffix>.parquet and keeps outputs separate per scale.
#   SCALE=500k sbatch --array=0-4 ...   (default)
#   SCALE=1M   sbatch --array=0-4 ...
SCALE="${SCALE:-500k}"

# Experiment -> EU/English ratio. Suffix (e.g. 100en) selects the assembled file.
EXPERIMENTS=("G1-100en" "G2-75en" "G3-50en" "G4-25en" "G5-0en")
TASK_ID="${SLURM_ARRAY_TASK_ID:?Must run as array job: sbatch --array=0-4 ...}"
EXPERIMENT="${EXPERIMENTS[$TASK_ID]}"
EN_TAG="${EXPERIMENT#*-}"                       # G1-100en -> 100en

INPUT_PARQUET="$ASSEMBLED_DIR/synthif-${SCALE}-${EN_TAG}.parquet"
OUTPUT_DIR="$PROJECT_ROOT/data/datasets_multilingual_sft/tokenized/${SCALE}/${EXPERIMENT}"

# --- HF offline (shared cache holds the Olmo-3 tokenizer) ---------------------
HF_HOME="${HF_HOME:-$SCRATCH_ROOT/cache/huggingface/abhasjha}"
# The container only binds the /pfs path; an inherited /scratch/<proj> form (a
# symlink on the host) is invisible inside it -> normalize to the canonical path.
case "$HF_HOME" in
  /scratch/project_462001516/*) HF_HOME="$SCRATCH_ROOT/${HF_HOME#/scratch/project_462001516/}" ;;
esac
export HF_HOME
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# Stop deepspeed (imported transitively by open_instruct) from auto-probing the
# accelerator at import, which calls amdsmi_init and can hang.
export DS_ACCELERATOR=cpu
# Stream stdout/stderr live instead of block-buffering it; otherwise the log
# looks frozen at import for hours even while tokenization is actively running.
export PYTHONUNBUFFERED=1
# CRITICAL: without this the fast (Rust) tokenizer initializes its threadpool,
# then datasets.map(num_proc=N) forks the worker processes and every child
# deadlocks -- workers spin at 100% CPU forever, emit no progress, and the job
# dies at the wall limit with zero output (observed: jobs 19457647/19457711 all
# TIMEOUT'd at 12h). Disabling Rust-side parallelism makes the fork safe and lets
# the process-level map workers do the work (~minutes for 500k, not hours).
export TOKENIZERS_PARALLELISM=false

# convert_sft_data_for_olmocore.py reads this to size its worker pool.
export BEAKER_ASSIGNED_CPU_COUNT=$(( ${SLURM_CPUS_PER_TASK:-32} * 3 / 4 ))

mkdir -p "$OUTPUT_DIR" "$PROJECT_ROOT/oellm/pipelines/tokenization/logs"
[ -f "$INPUT_PARQUET" ] || { echo "ERROR: $INPUT_PARQUET not found"; exit 1; }

echo "=============================================="
echo "Track G Tokenization (LUMI): $EXPERIMENT  [scale=$SCALE]"
echo "=============================================="
echo "Input:     $INPUT_PARQUET"
echo "Output:    $OUTPUT_DIR"
echo "Tokenizer: $TOKENIZER"
echo "Workers:   $BEAKER_ASSIGNED_CPU_COUNT"
echo "=============================================="

# Singularity inherits the exported env above (HF_*, BEAKER_*), so the inner
# shell only needs to activate the container's conda env + the layering venv.
singularity exec \
  --bind "$SCRATCH_ROOT:$SCRATCH_ROOT" \
  "$SIF" bash -c "
    set -euo pipefail
    source /opt/miniconda3/bin/activate pytorch
    source '$VENV/bin/activate'
    # ray (imported transitively by open_instruct) rejects LUMI's
    # ROCR_VISIBLE_DEVICES and demands HIP_VISIBLE_DEVICES -- translate it.
    if [ -n \"\${ROCR_VISIBLE_DEVICES:-}\" ]; then
        export HIP_VISIBLE_DEVICES=\"\$ROCR_VISIBLE_DEVICES\"
        unset ROCR_VISIBLE_DEVICES
    fi
    cd '$PROJECT_ROOT'
    # --dataset_skip_cache avoids DatasetTransformationCache, which otherwise
    # calls hf_whoami() (an ungated network request, not respecting
    # HF_HUB_OFFLINE) to resolve a default hf_entity -- this hangs forever on
    # a no-internet compute node instead of failing fast.
    python scripts/data/convert_sft_data_for_olmocore.py \
        --tokenizer_name_or_path '$TOKENIZER' \
        --dataset_mixer_list '$INPUT_PARQUET' 1.0 \
        --output_dir '$OUTPUT_DIR' \
        --chat_template_name olmo \
        --max_seq_length $MAX_SEQ_LENGTH \
        --dataset_skip_cache \
        --visualize
  "

echo "TOKENIZATION COMPLETE: $EXPERIMENT"
