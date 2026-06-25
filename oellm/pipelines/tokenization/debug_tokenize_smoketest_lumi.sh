#!/usr/bin/env bash
# Confirmation smoke test for the tokenizers fork-deadlock fix.
#
# Slices ~5k rows from the 500k-100en assembled mixture and runs the OLMo-core
# converter with TOKENIZERS_PARALLELISM=false. If the deadlock hypothesis is
# right, this finishes in a couple of minutes (emitting `datasets` Map progress
# and writing .npy output) instead of spinning at 100% CPU forever.
#
# Run (interactive, foreground so you watch it live):
#   srun --account=project_465002530 --partition=dev-g --nodes=1 --ntasks=1 \
#        --cpus-per-task=32 --mem=120G --time=00:20:00 \
#        oellm/pipelines/tokenization/debug_tokenize_smoketest_lumi.sh 2>&1 \
#        | tee oellm/pipelines/tokenization/logs/smoketest_$(date +%Y%m%d_%H%M%S).log

set -euo pipefail

SCRATCH_ROOT=/pfs/lustrep1/scratch/project_462001516
LUMI_DIR="$SCRATCH_ROOT/abhasjha/lumi-container"
SIF="$LUMI_DIR/lumi-pytorch-rocm-6.2.1-python-3.12-pytorch-20240918-vllm-4075b35.sif"
VENV="$LUMI_DIR/venv"
PROJECT_ROOT="$SCRATCH_ROOT/abhasjha/fabio-open-instruct/open-instruct"
ASSEMBLED_DIR="$SCRATCH_ROOT/abhasjha/gen_outputs/outputs/open_instruct/assembled"
TOKENIZER="allenai/Olmo-3-7B-Instruct-SFT"
MAX_SEQ_LENGTH=32768
N_ROWS="${N_ROWS:-5000}"

FULL_PARQUET="$ASSEMBLED_DIR/synthif-500k-100en.parquet"
SLICE_PARQUET="$SCRATCH_ROOT/abhasjha/tmp/smoketest_slice_${N_ROWS}.parquet"
OUTPUT_DIR="$PROJECT_ROOT/data/datasets_multilingual_sft/tokenized/_smoketest/${N_ROWS}rows"

# --- HF offline ---------------------------------------------------------------
HF_HOME="${HF_HOME:-$SCRATCH_ROOT/cache/huggingface/abhasjha}"
case "$HF_HOME" in
  /scratch/project_462001516/*) HF_HOME="$SCRATCH_ROOT/${HF_HOME#/scratch/project_462001516/}" ;;
esac
export HF_HOME
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DS_ACCELERATOR=cpu
export PYTHONUNBUFFERED=1
# THE FIX UNDER TEST: stop the fast tokenizer's Rust threadpool so datasets.map
# can fork worker processes without deadlocking.
export TOKENIZERS_PARALLELISM=false
export BEAKER_ASSIGNED_CPU_COUNT=$(( ${SLURM_CPUS_PER_TASK:-32} * 3 / 4 ))

mkdir -p "$OUTPUT_DIR" "$(dirname "$SLICE_PARQUET")"

echo "=============================================="
echo "Tokenize smoke test (LUMI): ${N_ROWS} rows of synthif-500k-100en"
echo "  TOKENIZERS_PARALLELISM=$TOKENIZERS_PARALLELISM  workers=$BEAKER_ASSIGNED_CPU_COUNT"
echo "  output=$OUTPUT_DIR"
echo "=============================================="

singularity exec --bind "$SCRATCH_ROOT:$SCRATCH_ROOT" "$SIF" bash -c "
  set -euo pipefail
  source /opt/miniconda3/bin/activate pytorch
  source '$VENV/bin/activate'
  if [ -n \"\${ROCR_VISIBLE_DEVICES:-}\" ]; then
      export HIP_VISIBLE_DEVICES=\"\$ROCR_VISIBLE_DEVICES\"; unset ROCR_VISIBLE_DEVICES
  fi
  cd '$PROJECT_ROOT'

  # Build the small slice once (skip if it already exists).
  if [ ! -f '$SLICE_PARQUET' ]; then
    echo '[slice] writing $N_ROWS-row slice...'
    python -c \"
import pyarrow.parquet as pq, pyarrow as pa
pf = pq.ParquetFile('$FULL_PARQUET')
b = next(pf.iter_batches(batch_size=$N_ROWS))
pq.write_table(pa.Table.from_batches([b]), '$SLICE_PARQUET')
print('[slice] wrote', '$SLICE_PARQUET', b.num_rows, 'rows')
\"
  fi

  echo '[convert] starting converter...'
  time python scripts/data/convert_sft_data_for_olmocore.py \
      --tokenizer_name_or_path '$TOKENIZER' \
      --dataset_mixer_list '$SLICE_PARQUET' 1.0 \
      --output_dir '$OUTPUT_DIR' \
      --chat_template_name olmo \
      --max_seq_length $MAX_SEQ_LENGTH \
      --dataset_skip_cache \
      --visualize
"

echo "=============================================="
echo "SMOKE TEST DONE. Output contents:"
ls -la "$OUTPUT_DIR"
echo "=============================================="
