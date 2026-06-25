#!/usr/bin/env bash
#SBATCH --job-name=tokenize-G
#SBATCH --partition=boost_usr_prod
#SBATCH --account=OELLM_prod2026
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --output=/leonardo_work/OELLM_prod2026/users/ajha0001/fabio-open-instruct/open-instruct/oellm/pipelines/tokenization/logs/tokenize_trackG_%A_%a.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=abhash.kumar.jha@tue.ellis.eu

# Tokenize Track G 500k assembled mixtures (synthetic-IF EU + Dolci English replay)
# into OLMo-core numpy layout. Leonardo Booster port of tokenize_trackG.sh.
#
# A GPU is requested only because boost_usr_prod requires one; tokenization is
# CPU-bound (HF tokenizer multiprocessing). Compute nodes have no internet, so
# the Olmo-3 tokenizer is read from the shared HF cache in offline mode.
#
# Usage: sbatch --array=0-4 oellm/pipelines/tokenization/tokenize_trackG_leonardo.sh

set -euo pipefail

PROJECT_ROOT="/leonardo_work/OELLM_prod2026/users/ajha0001/fabio-open-instruct/open-instruct"
ASSEMBLED_DIR="/leonardo_work/OELLM_prod2026/users/ajha0001/synthetic_generation/pilot/outputs/open_instruct/assembled"
TOKENIZER="allenai/Olmo-3-7B-Instruct-SFT"
MAX_SEQ_LENGTH=32768

# Experiment -> EU/English ratio. Suffix (e.g. 100en) selects the assembled file
# synthif-500k-<suffix>.parquet.
EXPERIMENTS=("G1-100en" "G2-75en" "G3-50en" "G4-25en" "G5-0en")
TASK_ID="${SLURM_ARRAY_TASK_ID:?Must run as array job: sbatch --array=0-4 ...}"
EXPERIMENT="${EXPERIMENTS[$TASK_ID]}"
EN_TAG="${EXPERIMENT#*-}"                       # G1-100en -> 100en

INPUT_PARQUET="$ASSEMBLED_DIR/synthif-500k-${EN_TAG}.parquet"
OUTPUT_DIR="$PROJECT_ROOT/data/datasets_multilingual_sft/tokenized/${EXPERIMENT}"

# --- HF offline (shared cache holds the Olmo-3 tokenizer) -------------------
export HF_HOME="${HF_HOME:-/leonardo_work/OELLM_prod2026/ytahtah0/.cache/huggingface}"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# convert_sft_data_for_olmocore.py reads this to size its worker pool.
WORKER_COUNT=$((${SLURM_CPUS_PER_TASK:-32} * 3 / 4))
export BEAKER_ASSIGNED_CPU_COUNT="$WORKER_COUNT"

# --- CUDA forward-compat (host driver caps at 12.2) ------------------------
module load cuda/12.6 2>/dev/null || true
if [ -n "${CUDA_HOME:-}" ] && [ -d "$CUDA_HOME/compat" ]; then
    export LD_LIBRARY_PATH="$CUDA_HOME/compat:${LD_LIBRARY_PATH:-}"
fi

source "$PROJECT_ROOT/.venv/bin/activate"
mkdir -p "$OUTPUT_DIR" "$PROJECT_ROOT/oellm/pipelines/tokenization/logs"

echo "=============================================="
echo "Track G Tokenization (Leonardo): $EXPERIMENT"
echo "=============================================="
echo "Input:     $INPUT_PARQUET"
echo "Output:    $OUTPUT_DIR"
echo "Tokenizer: $TOKENIZER"
echo "Workers:   $BEAKER_ASSIGNED_CPU_COUNT"
echo "=============================================="

[ -f "$INPUT_PARQUET" ] || { echo "ERROR: $INPUT_PARQUET not found"; exit 1; }

python "$PROJECT_ROOT/scripts/data/convert_sft_data_for_olmocore.py" \
    --tokenizer_name_or_path "$TOKENIZER" \
    --dataset_mixer_list "$INPUT_PARQUET" 1.0 \
    --output_dir "$OUTPUT_DIR" \
    --chat_template_name olmo \
    --max_seq_length "$MAX_SEQ_LENGTH" \
    --visualize

echo "TOKENIZATION COMPLETE: $EXPERIMENT"
