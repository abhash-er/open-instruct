#!/usr/bin/env bash
#SBATCH --job-name=hf2olmocore
#SBATCH --partition=boost_usr_prod
#SBATCH --account=OELLM_prod2026
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=/leonardo_work/OELLM_prod2026/users/ajha0001/fabio-open-instruct/open-instruct/oellm/pipelines/preprocessing/logs/hf2olmocore_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=abhash.kumar.jha@tue.ellis.eu

# Convert the HF allenai/Olmo-3-7B-Instruct-SFT checkpoint into OLMo-core format
# so OLMo-sft.py can load it as the pretraining checkpoint for Track G SFT.
# One-shot; produces the shared base checkpoint for all five G* runs.

set -euo pipefail

PROJECT_ROOT="/leonardo_work/OELLM_prod2026/users/ajha0001/fabio-open-instruct/open-instruct"
OLMOCORE_PATH="/leonardo_work/OELLM_prod2026/users/ajha0001/fabio-open-instruct/OLMo-core"
HF_MODEL="allenai/Olmo-3-7B-Instruct-SFT"
OUTPUT_DIR="${PROJECT_ROOT}/checkpoints/base/Olmo-3-7B-Instruct-SFT-olmocore"

# HF offline cache (shared) — compute nodes have no internet.
export HF_HOME="${HF_HOME:-/leonardo_work/OELLM_prod2026/ytahtah0/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# CUDA forward-compat (host driver caps at 12.2).
module load cuda/12.6 2>/dev/null || true
if [ -n "${CUDA_HOME:-}" ] && [ -d "$CUDA_HOME/compat" ]; then
    export LD_LIBRARY_PATH="$CUDA_HOME/compat:${LD_LIBRARY_PATH:-}"
fi

source "$PROJECT_ROOT/.venv/bin/activate"
# Resolve `import olmo_core` to the team fork (has the SFT + conversion code).
export PYTHONPATH="${OLMOCORE_PATH}/src:${PYTHONPATH:-}"

mkdir -p "$OUTPUT_DIR" "$(dirname "$PROJECT_ROOT/oellm/pipelines/preprocessing/logs")/logs" \
         "$PROJECT_ROOT/oellm/pipelines/preprocessing/logs"

echo "=============================================="
echo "HF -> OLMo-core conversion"
echo "  HF model : $HF_MODEL"
echo "  Arch     : olmo3_7b   Tokenizer: dolma2"
echo "  Output   : $OUTPUT_DIR"
echo "=============================================="

python "${OLMOCORE_PATH}/src/examples/huggingface/convert_checkpoint_from_hf.py" \
    --checkpoint-input-path "$HF_MODEL" \
    --model-arch olmo3_7b \
    --tokenizer dolma2 \
    --output-dir "$OUTPUT_DIR" \
    --device cuda

echo "CONVERSION COMPLETE -> $OUTPUT_DIR"
