#!/usr/bin/env bash
#SBATCH --job-name=synthif-sft
#SBATCH --account=project_465002530
#SBATCH --partition=standard-g
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=/pfs/lustrep1/scratch/project_462001516/abhasjha/fabio-open-instruct/open-instruct/oellm/pipelines/training/logs/%x_%j.out
#SBATCH --error=/pfs/lustrep1/scratch/project_462001516/abhasjha/fabio-open-instruct/open-instruct/oellm/pipelines/training/logs/%x_%j.err

# LUMI-G SFT training for OLMo-3-7B on the synthif mixtures (Track G).
# Port of train_multilingual_sft_horeka_a100.sh. Runs OLMo-sft.py from the
# ferreirafabio/OLMo-core (sft-slurm) fork inside the LUMI ROCm container.
#
# olmo_core inits its own distributed process group (FSDP, RCCL backend) from
# the standard torch env vars, so we launch ONE srun task per GCD (8/node) and
# feed each task RANK/LOCAL_RANK/WORLD_SIZE/MASTER_ADDR. No accelerate/deepspeed
# (deepspeed isn't in the container, and olmo_core doesn't use it).
#
# Usage (production, 2 nodes x 8 GCD = 16):
#   EXPERIMENT=G1-100en sbatch oellm/pipelines/training/train_synthif_sft_lumi.sh
#   EXPERIMENT=G3-50en NUM_NODES_HINT=4 sbatch -N4 oellm/pipelines/training/train_synthif_sft_lumi.sh
# Smoke test (5 steps) -> use test_train_5step_lumi.sh, which sets TEST_RUN/MAX_STEPS.

set -euo pipefail

# === paths (filesystem stays under project_462001516) =========================
SCRATCH=/pfs/lustrep1/scratch/project_462001516
LUMI_DIR="$SCRATCH/abhasjha/lumi-container"
PROJECT_ROOT="$SCRATCH/abhasjha/fabio-open-instruct/open-instruct"
OLMOCORE_PATH="$SCRATCH/abhasjha/fabio-open-instruct/OLMo-core"
OLMO_SFT="$OLMOCORE_PATH/src/scripts/train/sft/OLMo-sft.py"

# === container selection ======================================================
# Two interchangeable execution environments (see oellm/pipelines/container/README):
#   default            -> base SIF + the --system-site-packages overlay venv (enter.sh).
#                         Validated path; needs the conda+venv activation below.
#   USE_TRACKG_SIF=1   -> the self-contained cotainr image (lumi-trackg-train.sif): the
#                         whole stack is baked + auto-activated, so NO venv/conda source,
#                         but the aws-ofi-rccl plugin (absent from the rocm-6.2 base) must
#                         be bound in for RCCL-over-Slingshot.
VENV="$LUMI_DIR/venv"
if [[ "${USE_TRACKG_SIF:-0}" == "1" ]]; then
  SIF="${TRACKG_SIF:-$LUMI_DIR/lumi-trackg-train.sif}"
  ACTIVATE_BLOCK=":  # self-contained image: conda env is baked + auto-activated"
  AWS_OFI_BIND="--bind $LUMI_DIR/aws-ofi-rccl:/opt/aws-ofi-rccl"
else
  SIF="$LUMI_DIR/lumi-pytorch-rocm-6.2.1-python-3.12-pytorch-20240918-vllm-4075b35.sif"
  ACTIVATE_BLOCK="source /opt/miniconda3/bin/activate pytorch; source \"$VENV/bin/activate\""
  AWS_OFI_BIND=""
fi

# === experiment config ========================================================
EXPERIMENT="${EXPERIMENT:?Usage: EXPERIMENT=G1-100en sbatch train_synthif_sft_lumi.sh}"
RUN_NAME="${RUN_NAME:-synthif-${EXPERIMENT}}"
CLUSTER_NAME="slurm"   # not in OLMo-sft's Beaker map -> it infers GPU type from the device

SCALE="${SCALE:-500k}"   # which tokenized scale to train on (matches tokenize_trackG_lumi.sh)
DATASET_PATH="${DATASET_PATH:-${PROJECT_ROOT}/data/datasets_multilingual_sft/tokenized/${SCALE}/${EXPERIMENT}}"
# Point at the model_and_optim subdir, not its parent: olmo_core's load_checkpoint
# gates on dir_is_checkpoint(), which only accepts a dir with a top-level `.metadata`
# (our DCP metadata lives at model_and_optim/.metadata) or a full trainer checkpoint
# (train/rank0.pt + .metadata.json -- which a converted base model has not). The
# parent dir matches neither -> "No checkpoints found". model_and_optim/ has the
# top-level .metadata, so dir_is_checkpoint passes and load() reads the DCP there.
BASE_CKPT="${BASE_CKPT:-${PROJECT_ROOT}/checkpoints/base/Olmo-3-7B-Instruct-SFT-olmocore/model_and_optim}"

# === hyperparameters (OLMo-3 SFT recipe; override via env) ====================
LEARNING_RATE="${LEARNING_RATE:-8e-5}"
SEQ_LEN="${SEQ_LEN:-32768}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$((SEQ_LEN * 32))}"   # ~1M tokens
# Per-rank microbatch cap (passed as --max_rank_microbatch_size_tokens, same as the
# Horeka scripts). At 16384 = SEQ_LEN/2 it forces cp_degree=2, so each 32768-token
# sequence is split across 2 GCDs. This is REQUIRED on MI250X: at 32768 (cp_degree=1)
# the smallest possible microbatch is one full 32768-token sequence, which OOMs the
# 64GB GCD (~54GB). cp_degree=2 ~halves activation memory (the Horeka A100-80GB runs
# could afford cp_degree=1; the MI250X cannot).
MAX_RANK_MICROBATCH_SIZE_TOKENS="${MAX_RANK_MICROBATCH_SIZE_TOKENS:-16384}"
SAVE_INTERVAL="${SAVE_INTERVAL:-200}"

# === duration / smoke-test knobs ==============================================
TEST_RUN="${TEST_RUN:-false}"
MAX_STEPS="${MAX_STEPS:-5}"
EXTRA_OVERRIDES=""
if [[ "$TEST_RUN" == "true" ]]; then
  echo "TEST_RUN=true -> limiting to ${MAX_STEPS} steps."
  MAX_DURATION_VALUE="$MAX_STEPS"; MAX_DURATION_UNIT="steps"
  EXTRA_OVERRIDES="--trainer.callbacks.checkpointer.save_interval=${MAX_STEPS} --trainer.callbacks.checkpointer.ephemeral_save_interval=$((MAX_STEPS > 1 ? MAX_STEPS/2 : 1))"
else
  MAX_DURATION_VALUE="2"; MAX_DURATION_UNIT="epochs"
fi

# === HF cache: normalize an inherited /scratch/<proj> path to the canonical /pfs
# form, otherwise it's invisible inside the container (only /pfs is bound). =====
HF_HOME="${HF_HOME:-$SCRATCH/cache/huggingface/abhasjha}"
case "$HF_HOME" in
  /scratch/project_462001516/*) HF_HOME="$SCRATCH/${HF_HOME#/scratch/project_462001516/}" ;;
esac
export HF_HOME

# === topology (read from the live allocation; works for sbatch and salloc) ====
NUM_NODES="${SLURM_NNODES:-1}"
GPUS_PER_NODE=8
TOTAL_GPUS=$((GPUS_PER_NODE * NUM_NODES))
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
export MASTER_PORT=$((29500 + (${SLURM_JOB_ID:-0} % 10000)))

# === W&B (off unless WANDB_API_KEY is set) ====================================
WANDB_ENABLED="${WANDB_ENABLED:-auto}"
if [[ "$WANDB_ENABLED" == "auto" ]]; then
  [[ -n "${WANDB_API_KEY:-}" ]] && WANDB_ENABLED=true || WANDB_ENABLED=false
fi
WANDB_PROJECT="${WANDB_PROJECT:-olmo-synthif-sft}"
WANDB_ENTITY="${WANDB_ENTITY:-}"

mkdir -p "$PROJECT_ROOT/oellm/pipelines/training/logs"

echo "=============================================="
echo "LUMI synthif SFT  |  EXPERIMENT=$EXPERIMENT  RUN_NAME=$RUN_NAME"
echo "  nodes=$NUM_NODES  gcd/node=$GPUS_PER_NODE  world=$TOTAL_GPUS"
echo "  dataset=$DATASET_PATH"
echo "  base_ckpt=$BASE_CKPT"
echo "  seq_len=$SEQ_LEN  gbs=$GLOBAL_BATCH_SIZE  microbatch=$MAX_RANK_MICROBATCH_SIZE_TOKENS  lr=$LEARNING_RATE"
echo "  duration=$MAX_DURATION_VALUE $MAX_DURATION_UNIT  master=$MASTER_ADDR:$MASTER_PORT"
echo "=============================================="
[[ -d "$DATASET_PATH" ]]  || { echo "ERROR: dataset not found: $DATASET_PATH (tokenize first)"; exit 1; }
[[ -e "$BASE_CKPT"   ]]   || { echo "ERROR: base ckpt not found: $BASE_CKPT (convert first)"; exit 1; }

# === per-rank launcher (one task per GCD) =====================================
# Written to a temp file to keep the srun -> singularity -> python nesting sane.
# MUST live on shared Lustre (not node-local /tmp): srun launches this on every
# node, and a /tmp path only exists on the batch node -> the other nodes fail
# with `execve(): ... No such file or directory`.
RUNNER_DIR="$PROJECT_ROOT/oellm/pipelines/training/logs/.runners"
mkdir -p "$RUNNER_DIR"
RUNNER="$(mktemp "$RUNNER_DIR/synthif_runner.XXXXXX.sh")"
trap 'rm -f "$RUNNER"' EXIT
cat > "$RUNNER" <<EOF
#!/usr/bin/env bash
set -euo pipefail
# torch distributed env, derived per-task from SLURM
export RANK=\$SLURM_PROCID
export LOCAL_RANK=\$SLURM_LOCALID
export WORLD_SIZE=\$SLURM_NTASKS
export LOCAL_WORLD_SIZE=\${SLURM_NTASKS_PER_NODE:-$GPUS_PER_NODE}
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT
# all 8 GCDs visible to every task; olmo_core does torch.cuda.set_device(LOCAL_RANK)
unset ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
# per-rank MIOpen cache (avoid cross-rank contention on the shared FS)
export MIOPEN_USER_DB_PATH="/tmp/miopen-\${SLURM_NODEID}-\${SLURM_LOCALID}"
export MIOPEN_CUSTOM_CACHE_DIR="\$MIOPEN_USER_DB_PATH"
mkdir -p "\$MIOPEN_USER_DB_PATH"

singularity exec \
  --bind "$SCRATCH:$SCRATCH" \
  --bind /opt/cray:/opt/cray \
  --bind /var/spool/slurmd:/var/spool/slurmd \
  --bind /usr/lib64/libcxi.so.1:/usr/lib64/libcxi.so.1 \
  --bind /usr/lib64/libjansson.so.4:/usr/lib64/libjansson.so.4 \
  $AWS_OFI_BIND \
  "$SIF" bash -c '
    set -euo pipefail
    $ACTIVATE_BLOCK
    # RCCL over Slingshot via the aws-ofi-rccl plugin (bundled in the base SIF; bound
    # in from scratch for the self-contained trackg image)
    export LD_LIBRARY_PATH=/opt/aws-ofi-rccl:/opt/cray/lib64:\${LD_LIBRARY_PATH:-}
    export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
    export NCCL_NET_GDR_LEVEL=PHB
    export NCCL_DEBUG=\${NCCL_DEBUG:-WARN}
    export OLMO_SHARED_FS=1
    export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
    export PYTHONPATH="$OLMOCORE_PATH/src:\${PYTHONPATH:-}"
    cd "$PROJECT_ROOT"
    python "$OLMO_SFT" train \
        "$RUN_NAME" \
        "$BASE_CKPT" \
        "$CLUSTER_NAME" \
        --seq_len=$SEQ_LEN \
        --num_nodes=$NUM_NODES \
        --global_batch_size=$GLOBAL_BATCH_SIZE \
        --max_rank_microbatch_size_tokens=$MAX_RANK_MICROBATCH_SIZE_TOKENS \
        --model_name=olmo3-7b \
        --dataset_path="$DATASET_PATH" \
        --train_module.optim.lr=$LEARNING_RATE \
        --trainer.max_duration.value=$MAX_DURATION_VALUE \
        --trainer.max_duration.unit=$MAX_DURATION_UNIT \
        --trainer.callbacks.checkpointer.ephemeral_save_interval=$SAVE_INTERVAL \
        --trainer.callbacks.wandb.enabled=$WANDB_ENABLED \
        --trainer.callbacks.wandb.project=$WANDB_PROJECT \
        --trainer.callbacks.wandb.entity=$WANDB_ENTITY \
        --trainer.callbacks.wandb.name="$RUN_NAME" \
        --save_tokenizer=True \
        --budget=unused \
        --workspace=unused \
        $EXTRA_OVERRIDES
  '
EOF
chmod +x "$RUNNER"

# NUMA-local CPU binding for the 8 GCDs (LUMI-G standard mask).
CPU_BIND_MASK="0x00fe000000000000,0xfe00000000000000,0x0000000000fe0000,0x00000000fe000000,0x00000000000000fe,0x000000000000fe00,0x000000fe00000000,0x0000fe0000000000"

srun --cpu-bind="mask_cpu:$CPU_BIND_MASK" "$RUNNER"

echo "TRAINING FINISHED: $RUN_NAME"
