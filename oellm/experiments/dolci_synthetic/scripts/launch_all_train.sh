#!/usr/bin/env bash
# Launch all Track-G (dolci-synthetic) SFT runs on LUMI-G.
#
# Track G fine-tunes the OLMo-3-7B-Instruct-SFT base on the synthif mixtures:
# synthetic instruction-following (EU langs) + Dolci English replay, swept across
# five English ratios (G1-100en .. G5-0en) at two scales (500k, 1M) = 10 runs.
#
# This is a thin orchestration layer over the generic training pipeline at
# oellm/pipelines/training/train_synthif_sft_lumi.sh -- it just submits one
# sbatch per (experiment, scale) cell with the right env. Two concerns it owns
# that the generic script can't:
#   1. RUN_NAME / save folder must include the SCALE. The pipeline's default
#      RUN_NAME is synthif-<EXP> (no scale), so 500k and 1M of the same mixture
#      would otherwise collide on one checkpoint dir. We set OLMO_SAVE_FOLDER
#      (read by OLMo-sft.py) and RUN_NAME per cell to keep them separate.
#   2. Checkpoints land under this experiment's tree, not the OLMo-core default.
#
# Usage:
#   # everything (10 runs):
#   ./oellm/experiments/dolci_synthetic/scripts/launch_all_train.sh
#   # one scale:
#   SCALES="500k" ./...launch_all_train.sh
#   # subset of mixtures:
#   EXPERIMENTS="G1-100en G5-0en" SCALES="1M" ./...launch_all_train.sh
#   # dry run (print sbatch lines, submit nothing):
#   DRY_RUN=1 ./...launch_all_train.sh

set -euo pipefail

SCRATCH=/pfs/lustrep1/scratch/project_462001516
PROJECT_ROOT="$SCRATCH/abhasjha/fabio-open-instruct/open-instruct"
TRAIN_SCRIPT="$PROJECT_ROOT/oellm/pipelines/training/train_synthif_sft_lumi.sh"
CKPT_ROOT="${CKPT_ROOT:-$PROJECT_ROOT/checkpoints/synthif}"

EXPERIMENTS="${EXPERIMENTS:-G1-100en G2-75en G3-50en G4-25en G5-0en}"
SCALES="${SCALES:-500k 1M}"
DRY_RUN="${DRY_RUN:-0}"

[ -f "$TRAIN_SCRIPT" ] || { echo "ERROR: train script not found: $TRAIN_SCRIPT"; exit 1; }

echo "=============================================="
echo "Track G (dolci-synthetic) -- launch all SFT runs"
echo "  experiments: $EXPERIMENTS"
echo "  scales:      $SCALES"
echo "  ckpt root:   $CKPT_ROOT"
echo "  dry run:     $DRY_RUN"
echo "=============================================="

for SCALE in $SCALES; do
  for EXP in $EXPERIMENTS; do
    DATASET_PATH="$PROJECT_ROOT/data/datasets_multilingual_sft/tokenized/${SCALE}/${EXP}"
    if [ ! -d "$DATASET_PATH" ]; then
      echo "SKIP ${SCALE}/${EXP}: tokenized data missing ($DATASET_PATH) -- tokenize first."
      continue
    fi

    RUN_NAME="synthif-${SCALE}-${EXP}"
    SAVE_FOLDER="$CKPT_ROOT/${RUN_NAME}"

    echo
    echo ">>> submit ${RUN_NAME}"
    CMD=(sbatch --job-name="$RUN_NAME"
         "$TRAIN_SCRIPT")
    # Env consumed by train_synthif_sft_lumi.sh (EXPERIMENT/SCALE/RUN_NAME) and by
    # OLMo-sft.py (OLMO_SAVE_FOLDER). sbatch --export=ALL (default) carries them in.
    ENV=(EXPERIMENT="$EXP" SCALE="$SCALE" RUN_NAME="$RUN_NAME"
         OLMO_SAVE_FOLDER="$SAVE_FOLDER"
         WANDB_PROJECT="${WANDB_PROJECT:-olmo-synthif-sft}")

    if [ "$DRY_RUN" = "1" ]; then
      echo "    DRY: ${ENV[*]} ${CMD[*]}"
    else
      env "${ENV[@]}" "${CMD[@]}"
    fi
  done
done

echo
echo "Done. Track jobs with: squeue --me"
