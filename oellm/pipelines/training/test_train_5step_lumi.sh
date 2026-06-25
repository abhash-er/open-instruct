#!/usr/bin/env bash
#SBATCH --job-name=synthif-5step
#SBATCH --account=project_465002530
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --output=/pfs/lustrep1/scratch/project_462001516/abhasjha/fabio-open-instruct/open-instruct/oellm/pipelines/training/logs/%x_%j.out
#SBATCH --error=/pfs/lustrep1/scratch/project_462001516/abhasjha/fabio-open-instruct/open-instruct/oellm/pipelines/training/logs/%x_%j.err

# 5-step smoke test on dev-g (1 node x 8 GCD) to validate the env, RCCL/Slingshot
# init, checkpoint load and data load BEFORE committing to a full standard-g run.
# It just sets TEST_RUN/MAX_STEPS and hands off to train_synthif_sft_lumi.sh, so
# there's one source of truth for the actual launch.
#
# Two ways to use:
#   A) Batch:
#        EXPERIMENT=G1-100en sbatch oellm/pipelines/training/test_train_5step_lumi.sh
#   B) Interactive (grab a node, then run it inside the allocation):
#        salloc --account=project_465002530 --partition=dev-g --nodes=1 \
#               --gpus-per-node=8 --ntasks-per-node=8 --cpus-per-task=7 --time=00:30:00
#        EXPERIMENT=G1-100en ./oellm/pipelines/training/test_train_5step_lumi.sh
#
#   NOTE: in the interactive path use `salloc` (not `srun --pty bash`) -- the
#   training script issues its own `srun`, and running it inside an srun shell
#   would nest srun.

set -euo pipefail

export EXPERIMENT="${EXPERIMENT:-G1-100en}"
export TEST_RUN=true
export MAX_STEPS="${MAX_STEPS:-5}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"   # surface RCCL / aws-ofi-rccl init in the test log

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$HERE/train_synthif_sft_lumi.sh"
