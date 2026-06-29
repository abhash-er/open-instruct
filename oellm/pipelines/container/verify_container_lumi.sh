#!/usr/bin/env bash
# Sanity-check the baked Track-G training container WITHOUT sourcing any external
# venv/conda -- everything must come from inside lumi-trackg-train.sif.
#
# Checks (the exact things that broke during bring-up):
#   - torch 2.7.1 + ROCm (HIP) build, FSDP2 (FSDPModule) present
#   - flash_attn imports (no torch ABI error)
#   - ring_flash_attn imports incl. the patched (optional) HF-adapter path
#   - liger-kernel fused-linear-CE returns a 2-tuple (0.6.2, not 0.8.0's 4-tuple)
#   - omegaconf has to_object; transformers imports
#   - OLMo-core (bound fork via PYTHONPATH) imports its ROCm-patched bits
#
# Usage:
#   ./oellm/pipelines/container/verify_container_lumi.sh            # imports only (login node OK)
#   srun ... ./oellm/pipelines/container/verify_container_lumi.sh   # add a GPU for the fwd/bwd check
set -euo pipefail

SCRATCH=/pfs/lustrep1/scratch/project_462001516
LUMI_DIR="$SCRATCH/abhasjha/lumi-container"
SIF="${SIF:-$LUMI_DIR/lumi-trackg-train.sif}"
OLMOCORE_PATH="${OLMOCORE_PATH:-$SCRATCH/abhasjha/fabio-open-instruct/OLMo-core}"

[[ -f "$SIF" ]] || { echo "ERROR: container not found: $SIF (build it first)"; exit 1; }
echo "Verifying: $SIF"

singularity exec --bind "$SCRATCH:$SCRATCH" "$SIF" \
  bash -lc 'export PYTHONPATH="'"$OLMOCORE_PATH"'/src"; python - <<"PY"
import torch
print("torch:", torch.__version__, "| hip:", torch.version.hip, "| cuda:", torch.version.cuda)
assert torch.__version__.startswith("2.7.1"), torch.__version__
assert torch.version.hip, "expected a ROCm/HIP torch build"
from torch.distributed.fsdp import FSDPModule  # FSDP2 -- the reason we need torch>=2.7
print("FSDP2 FSDPModule: OK")

import flash_attn
print("flash_attn:", flash_attn.__version__)

import ring_flash_attn as rfa
from ring_flash_attn import zigzag_ring_flash_attn_func  # core dispatch (what OLMo-core uses)
print("ring_flash_attn: OK  (hf-adapter optional ->",
      "present" if rfa.substitute_hf_flash_attn is not None else "stubbed (patched)", ")")

import inspect
from liger_kernel.ops.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyFunction as L
src = inspect.getsource(L.forward)
ret = [l.strip() for l in src.splitlines() if l.strip().startswith("return")][-1]
print("liger forward return:", ret)
assert "token_accuracy" not in ret, "liger too new (4-tuple) -- need 0.6.2"
print("liger-kernel: OK (2-tuple)")

from omegaconf import OmegaConf
assert hasattr(OmegaConf, "to_object")
import transformers
print("omegaconf + transformers:", transformers.__version__, "OK")

import olmo_core
from olmo_core.train.utils import _get_cuda_version          # ROCm HIP-fallback patch
from olmo_core.nn.lm_head import LMLossImplementation        # fused_linear loss
print("olmo_core (bound fork): OK")
print("\nALL IMPORT CHECKS PASSED")
PY'

# Optional GPU fwd/bwd -- only runs if a visible device is present.
singularity exec --bind "$SCRATCH:$SCRATCH" "$SIF" bash -lc 'python - <<"PY"
import torch
if not torch.cuda.is_available():
    print("\n(no GPU visible -- skipping flash-attn fwd/bwd; rerun under srun for it)"); raise SystemExit
from flash_attn import flash_attn_func
q,k,v = (torch.randn(1,128,8,64, device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(3))
o = flash_attn_func(q,k,v, causal=True); o.sum().backward()
print("\nflash-attn fwd/bwd on", torch.cuda.get_device_name(0), ": OK", tuple(o.shape))
PY'
