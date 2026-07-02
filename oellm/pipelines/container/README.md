# Track-G LUMI training container

Two interchangeable ways to run the Track-G SFT pipeline on LUMI-G (MI250X). The
training script `oellm/pipelines/training/train_synthif_sft_lumi.sh` selects between
them with the `USE_TRACKG_SIF` env var. **The self-contained SIF is now the default**
(validated end-to-end on 2× MI250X — job 19623751); the venv overlay is a legacy opt-out.

| | **default** (self-contained SIF) | `USE_TRACKG_SIF=0` (legacy venv overlay) |
|---|---|---|
| Image | `lumi-trackg-train.sif` (cotainr) | base `lumi-pytorch-rocm-6.2.1-…-vllm-…sif` |
| Stack source | fully baked into the SIF | base conda `pytorch` env **+** `--system-site-packages` venv (`enter.sh`) |
| Needs uv/venv | **no** | yes (`enter.sh`) |
| vLLM | absent (training-only) | present (base image) |

Both run the OLMo-core **bound fork** via `PYTHONPATH=$OLMOCORE_PATH/src` (it holds the
ROCm/MI250X patches); the container only bakes the heavy/compiled deps.

## Why a separate training image (no vLLM)

OLMo-core SFT is pure FSDP2 training and imports no vLLM. It needs **torch ≥ 2.7**
(`torch.distributed.fsdp.FSDPModule` / FSDP2); baking torch 2.7.1 breaks the base image's
vLLM 0.6.3 and its prebuilt flash-attn (both built against torch 2.6). So this image is
training-only; **eval/inference/RL stays on the original base SIF** (torch 2.6 + vLLM).

## What's baked (validated stack)

torch 2.7.1+rocm6.2.4 (+ torchvision, pytorch-triton-rocm), our ROCm flash-attn 2.6.3
wheel (gfx90a), patched ring-flash-attn 0.1.8 (context parallelism), patched liger-kernel
0.6.2 (fused-linear CE), omegaconf 2.3.0, transformers 4.45.1, + olmo_core runtime deps.
Exact pins: `trackg-train-env.yml` (and `venv-pip-freeze.txt` for the venv overlay).

Three local wheels are **patched** and live in `wheels/` (gitignored — regenerate as below):
- `flash_attn-2.6.3-…` — compiled against torch 2.7.1+rocm6.2.4 (`build_flash_attn_lumi.sh`).
- `ring_flash_attn-0.1.8+lumipatch` — HF-adapter import made optional (it targets a
  transformers API this stack lacks; OLMo-core only uses the core ring dispatch funcs).
- `liger_kernel-0.6.2+notriton` — `triton` hard-dep stripped so a fresh resolve doesn't
  pull PyPI's CUDA triton and clobber torch's `pytorch-triton-rocm`.

## Build

Full from-scratch reproduction. Paths assume
`LUMI_DIR=/pfs/lustrep1/scratch/project_462001516/abhasjha/lumi-container` and the repo at
`$PROJECT_ROOT` (`.../fabio-open-instruct/open-instruct`). Run the `sbatch` steps from
anywhere; run the login-node steps on a **login node** (they need internet — compute nodes
have none). `singularity build`/fakeroot are unavailable for this user, so we use **cotainr**
(builds an unprivileged SIF from a conda-env spec).

### Prerequisites (one-off)

The flash-attn compile needs the source (incl. the `composable_kernel` submodule) cloned on
a login node, since the build runs on a no-internet compute node:

```bash
git clone https://github.com/Dao-AILab/flash-attention \
  "$LUMI_DIR/src/flash-attention"
cd "$LUMI_DIR/src/flash-attention" && git checkout v2.6.3 && \
  git submodule update --init --recursive csrc/composable_kernel
```

### Step 1 — flash-attn wheel (Slurm, CPU, ~1–2 h)

Rebuilds flash-attn 2.6.3 against torch 2.7.1+rocm6.2.4 for gfx90a (the base image's prebuilt
2.6.3 is ABI-incompatible with torch ≥ 2.7). CPU-only compile on the `small` partition.

```bash
sbatch oellm/pipelines/container/build_flash_attn_lumi.sh
# -> oellm/pipelines/container/wheels/flash_attn-2.6.3-cp312-cp312-linux_x86_64.whl
```

### Step 2 — patched pure-python wheels (login node, minutes)

Two upstream wheels need a small patch; both are pure-python, so "build" = download, edit,
repack. They live in `wheels/` (gitignored). Regenerate each by unzipping the upstream wheel,
making the edit, and re-zipping with a **local version tag** so it outranks PyPI:

- **`ring_flash_attn-0.1.8+lumipatch`** — make the HF-adapter import optional (it targets a
  `transformers` API this stack lacks; OLMo-core only uses the core ring dispatch funcs). Edit
  `ring_flash_attn/__init__.py` to wrap the `substitute_hf_flash_attn` import in `try/except`
  (set it to `None` on failure), bump `version` in `METADATA` to `0.1.8+lumipatch`, repack.
- **`liger_kernel-0.6.2+notriton`** — strip the `triton>=2.3.1` hard-dep so a fresh resolve
  doesn't pull PyPI's CUDA triton over torch's `pytorch-triton-rocm`. Delete the triton
  `Requires-Dist` line in `METADATA`, bump version to `0.6.2+notriton`, repack.

(See this dir's git history for the exact repack commands.)

### Step 3 — build the SIF (login node, internet, ~15 min)

```bash
./oellm/pipelines/container/build_container_lumi.sh   # -> $LUMI_DIR/lumi-trackg-train.sif
```

Uses `cotainr build --system rocm-6.2` (the LUMI ROCm 6.2.4 base image; `--system lumi-g` is a
broken symlink on this filesystem). cotainr's build sandbox runs `singularity exec --no-home`
with **no `/pfs` bind**, so it can't see the local wheels by path — the script works around this
by serving `wheels/` over `http://127.0.0.1:$WHEEL_PORT` (singularity shares the host network)
and substituting the port into a throwaway copy of `trackg-train-env.yml`. Everything else
(torch, torchvision, omegaconf, transformers, …) is pulled from PyPI / the pytorch ROCm index.

### Step 4 — extract the aws-ofi-rccl plugin (login node, one-off)

The fresh rocm-6.2 image lacks the base image's `/opt/aws-ofi-rccl` (RCCL-over-Slingshot). The
train script binds it in from `$LUMI_DIR/aws-ofi-rccl`; extract it once from the base SIF:

```bash
mkdir -p "$LUMI_DIR/aws-ofi-rccl"
singularity exec "$LUMI_DIR/lumi-pytorch-rocm-6.2.1-python-3.12-pytorch-20240918-vllm-4075b35.sif" \
  tar -C /opt/aws-ofi-rccl -cf - . | tar -C "$LUMI_DIR/aws-ofi-rccl" -xf -
```

### Step 5 — verify

```bash
# imports only (login node): torch 2.7.1+HIP, FSDP2, flash_attn, patched ring/liger, bound fork
./oellm/pipelines/container/verify_container_lumi.sh
# add a GPU for the flash-attn fwd/bwd check:
srun --account=project_462001516 --partition=dev-g --nodes=1 --gpus=1 --time=00:10:00 \
  ./oellm/pipelines/container/verify_container_lumi.sh
```

### Step 6 — smoke test (the real pass/fail signal)

20-step run on 2× MI250X through the training path itself (uses the SIF by default now):

```bash
EXPERIMENT=G1-100en RUN_NAME=synthif-sifcheck \
  sbatch oellm/pipelines/training/test_train_5step_lumi.sh
```

Expect `Training complete` + `step10/`/`step20/` checkpoints + ~50% MFU (matches the venv path).

## Run

```bash
# default: self-contained SIF (needs the aws-ofi-rccl plugin extracted to
# $LUMI_DIR/aws-ofi-rccl, bound in automatically):
EXPERIMENT=G1-100en sbatch --account=project_462001516 \
  oellm/pipelines/training/test_train_5step_lumi.sh

# legacy venv path (opt-out escape hatch):
USE_TRACKG_SIF=0 EXPERIMENT=G1-100en sbatch --account=project_462001516 \
  oellm/pipelines/training/test_train_5step_lumi.sh
```

The SIF lacks the base image's `/opt/aws-ofi-rccl` (RCCL-over-Slingshot plugin); the
train script binds it from `$LUMI_DIR/aws-ofi-rccl` (extract once from the base SIF:
`singularity exec base.sif tar -C /opt/aws-ofi-rccl -cf - . | tar -C $LUMI_DIR/aws-ofi-rccl -xf -`).
