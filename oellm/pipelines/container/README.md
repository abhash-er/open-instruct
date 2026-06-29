# Track-G LUMI training container

Two interchangeable ways to run the Track-G SFT pipeline on LUMI-G (MI250X). The
training script `oellm/pipelines/training/train_synthif_sft_lumi.sh` selects between
them with the `USE_TRACKG_SIF` env var.

| | default (venv overlay) | `USE_TRACKG_SIF=1` (self-contained) |
|---|---|---|
| Image | base `lumi-pytorch-rocm-6.2.1-…-vllm-…sif` | `lumi-trackg-train.sif` (cotainr) |
| Stack source | base conda `pytorch` env **+** `--system-site-packages` venv (`enter.sh`) | fully baked into the SIF |
| Needs uv/venv | yes (`enter.sh`) | **no** |
| vLLM | present (base image) | absent (training-only) |

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

```bash
# 1. flash-attn wheel (Slurm, CPU, one-off; needs the source cloned on a login node)
sbatch oellm/pipelines/container/build_flash_attn_lumi.sh
# 2. the SIF (login node — needs internet; ~15 min). Serves the local wheels over
#    localhost because cotainr's build sandbox can't see /pfs.
./oellm/pipelines/container/build_container_lumi.sh        # -> $LUMI_DIR/lumi-trackg-train.sif
# 3. verify (imports on login node; add a GPU via srun for the flash-attn fwd/bwd)
./oellm/pipelines/container/verify_container_lumi.sh
```

`build_container_lumi.sh` uses `cotainr --system rocm-6.2` (the LUMI ROCm 6.2.4 base;
`--system lumi-g` is a broken symlink on this filesystem). `singularity build`/fakeroot
are unavailable for this user, hence cotainr.

The ring/liger patched wheels are rebuilt by downloading the upstream wheel, editing
`__init__.py` / `METADATA`, and repacking (pure-python; see git history of this dir).

## Run

```bash
# default venv path (unchanged):
EXPERIMENT=G1-100en sbatch --account=project_462001516 \
  oellm/pipelines/training/test_train_5step_lumi.sh

# self-contained SIF (needs the aws-ofi-rccl plugin extracted to
# $LUMI_DIR/aws-ofi-rccl, bound in automatically):
USE_TRACKG_SIF=1 EXPERIMENT=G1-100en sbatch --account=project_462001516 \
  oellm/pipelines/training/test_train_5step_lumi.sh
```

The SIF lacks the base image's `/opt/aws-ofi-rccl` (RCCL-over-Slingshot plugin); the
train script binds it from `$LUMI_DIR/aws-ofi-rccl` (extract once from the base SIF:
`singularity exec base.sif tar -C /opt/aws-ofi-rccl -cf - . | tar -C $LUMI_DIR/aws-ofi-rccl -xf -`).
