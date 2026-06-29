#!/usr/bin/env bash
#SBATCH --job-name=build-flash-attn
#SBATCH --account=project_465002530
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=06:00:00
#SBATCH --output=/pfs/lustrep1/scratch/project_462001516/abhasjha/fabio-open-instruct/open-instruct/oellm/pipelines/container/logs/%x_%j.out
#SBATCH --error=/pfs/lustrep1/scratch/project_462001516/abhasjha/fabio-open-instruct/open-instruct/oellm/pipelines/container/logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=abhash.kumar.jha@tue.ellis.eu

# Compile a flash-attn wheel against torch 2.7.1+rocm6.2.4 for MI250X (gfx90a).
#
# WHY: the base container ships flash-attn 2.6.3 built against its conda torch
# 2.6.0.dev. Track-G training needs torch >= 2.7 (olmo_core FSDP2 / FSDPModule),
# and that torch upgrade breaks the prebuilt .so with an ABI error
# (`undefined symbol: ..._ZN3c105Error...`). Rebuilding the SAME 2.6.3 source
# against torch 2.7.1 produces a compatible wheel.
#
# Compute nodes have NO internet: the source (incl. the composable_kernel
# submodule) must already be cloned at $FA_SRC on a login node. The ROCm CK
# kernels compile for a while (tens of minutes to a couple hours for a single
# arch), hence the 6h wall and bounded MAX_JOBS to cap peak RAM.
#
# This is a CPU-only compile (GPU_ARCHS is set explicitly, nothing queries a live
# device), so it runs on the `small` LUMI-C partition -- no GPU reserved. The
# project (project_465002530) has CPU core-hours and `small` access. The separate
# import+fwd/bwd validation DOES need a GPU and runs elsewhere (small-g/dev-g).
#
# Usage:
#   sbatch oellm/pipelines/container/build_flash_attn_lumi.sh
# Output:
#   oellm/pipelines/container/wheels/flash_attn-*.whl

set -euo pipefail

SCRATCH=/pfs/lustrep1/scratch/project_462001516
LUMI_DIR="$SCRATCH/abhasjha/lumi-container"
SIF="$LUMI_DIR/lumi-pytorch-rocm-6.2.1-python-3.12-pytorch-20240918-vllm-4075b35.sif"
VENV="$LUMI_DIR/venv"   # provides torch 2.7.1+rocm6.2.4 (layered on conda python)
FA_SRC="${FA_SRC:-$LUMI_DIR/src/flash-attention}"
PROJECT_ROOT="$SCRATCH/abhasjha/fabio-open-instruct/open-instruct"
OUT_DIR="$PROJECT_ROOT/oellm/pipelines/container/wheels"

mkdir -p "$OUT_DIR" "$PROJECT_ROOT/oellm/pipelines/container/logs"
[[ -d "$FA_SRC/csrc/composable_kernel" ]] || {
  echo "ERROR: flash-attn source / composable_kernel submodule missing at $FA_SRC"
  echo "       clone it on a login node first (compute nodes have no internet):"
  echo "       git clone --depth 1 --branch v2.6.3 --recursive --shallow-submodules \\"
  echo "         https://github.com/Dao-AILab/flash-attention.git $FA_SRC"
  exit 1
}

echo "=============================================="
echo "flash-attn build | src=$FA_SRC | out=$OUT_DIR"
echo "=============================================="

singularity exec \
  --bind "$SCRATCH:$SCRATCH" \
  --bind /opt/cray:/opt/cray \
  "$SIF" bash -c '
    set -euo pipefail
    source /opt/miniconda3/bin/activate pytorch
    source "'"$VENV"'/bin/activate"

    # Build toolchain: ROCm hipcc (device) + a real host C/C++ compiler. This image
    # ships NO gcc/g++/c++, and torch'\''s extension ABI probe (`<cc> -v`, no input)
    # fails on clang/amdclang (they exit 1 "no input files"). LUMI'\''s gcc-mixed/12.2.0
    # provides self-contained gcc/g++ under /opt/cray (bound below) -- verified inside
    # this container: `g++ -v` exits 0 and it compiles+links cleanly.
    export ROCM_PATH=/opt/rocm-6.2.1
    export ROCM_HOME="$ROCM_PATH"
    export HIP_PATH="$ROCM_PATH"
    GNU_BIN=/opt/cray/pe/gcc/12.2.0/snos/bin
    [[ -x "$GNU_BIN/g++" ]] || { echo "ERROR: host gcc not found at $GNU_BIN (is /opt/cray bound?)"; exit 1; }
    export PATH="$GNU_BIN:$ROCM_PATH/bin:$PATH"
    export CC="$GNU_BIN/gcc"
    export CXX="$GNU_BIN/g++"

    # Single-arch build for MI250X keeps the compile tractable.
    export GPU_ARCHS=gfx90a
    export PYTORCH_ROCM_ARCH=gfx90a
    export FLASH_ATTENTION_FORCE_BUILD=TRUE
    # Cap parallel nvcc/hipcc jobs to bound peak RAM (CK TUs are memory-hungry).
    export MAX_JOBS="${MAX_JOBS:-24}"

    python -c "import torch; assert torch.version.hip, torch.__version__; print(\"build torch\", torch.__version__, torch.version.hip)"
    pip install --no-cache-dir --no-build-isolation ninja packaging wheel setuptools

    cd "'"$FA_SRC"'"
    echo "flash-attn source version: $(cat flash_attn/__init__.py | grep -m1 __version__ || true)"
    # Clean any stale build tree from a prior attempt.
    rm -rf build
    python setup.py bdist_wheel

    echo "=== built wheels ==="
    ls -la dist/
    cp -v dist/*.whl "'"$OUT_DIR"'/"
  '

echo "=============================================="
echo "DONE. Wheels in: $OUT_DIR"
ls -la "$OUT_DIR"
echo "=============================================="
