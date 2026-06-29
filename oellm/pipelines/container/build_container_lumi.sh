#!/usr/bin/env bash
# Build the Track-G LUMI *training* container (lumi-trackg-train.sif) with cotainr.
#
# TRAINING ONLY -- torch 2.7.1 + flash-attn + ring-flash-attn + liger, NO vLLM.
# (Baking torch 2.7.1 breaks the base image's vLLM 0.6.3 the same way it broke the
#  prebuilt flash-attn; vLLM/eval/inference stays on the original base SIF. See README.)
#
# RUN THIS ON A LOGIN NODE -- cotainr downloads conda + the torch ROCm wheels from
# the internet, which compute nodes cannot reach. No GPU is needed to build.
# fakeroot/`singularity build` are unavailable for this user, hence cotainr (which
# builds an unprivileged SIF from a conda env spec).
#
# Usage:
#   ./oellm/pipelines/container/build_container_lumi.sh
# Output:
#   $LUMI_DIR/lumi-trackg-train.sif
#
# Prereqs (already produced in this repo):
#   wheels/flash_attn-2.6.3-cp312-cp312-linux_x86_64.whl   (build_flash_attn_lumi.sh)
#   wheels/ring_flash_attn-0.1.8+lumipatch-py3-none-any.whl (patched, pure-python)
#   trackg-train-env.yml                                    (the conda env spec)

set -euo pipefail

SCRATCH=/pfs/lustrep1/scratch/project_462001516
LUMI_DIR="$SCRATCH/abhasjha/lumi-container"
PROJECT_ROOT="$SCRATCH/abhasjha/fabio-open-instruct/open-instruct"
CONTAINER_DIR="$PROJECT_ROOT/oellm/pipelines/container"
ENV_YML="$CONTAINER_DIR/trackg-train-env.yml"
SIF_OUT="${SIF_OUT:-$LUMI_DIR/lumi-trackg-train.sif}"

[[ -f "$ENV_YML" ]] || { echo "ERROR: env spec not found: $ENV_YML"; exit 1; }
for w in "$CONTAINER_DIR"/wheels/flash_attn-2.6.3-cp312-cp312-linux_x86_64.whl \
         "$CONTAINER_DIR"/wheels/ring_flash_attn-0.1.8+lumipatch-py3-none-any.whl; do
  [[ -f "$w" ]] || { echo "ERROR: required wheel missing: $w"; exit 1; }
done

# cotainr is exposed under the CrayEnv stack on LUMI.
module load CrayEnv
module load cotainr/2025.7.1

# --- serve the local wheels over localhost for the build -----------------------
# cotainr's build sandbox can't see /pfs (singularity exec --no-home, no bind), but
# it shares the host network, so we hand it the wheels over http and substitute the
# chosen port into a throwaway copy of the env yml (keeps the committed yml port-free).
WHEEL_PORT="${WHEEL_PORT:-8731}"
BUILT_ENV="$(mktemp /tmp/trackg-train-env.XXXXXX.yml)"
sed "s/__WHEEL_PORT__/$WHEEL_PORT/g" "$ENV_YML" > "$BUILT_ENV"

echo "Serving wheels on 127.0.0.1:$WHEEL_PORT from $CONTAINER_DIR/wheels"
( cd "$CONTAINER_DIR/wheels" && exec python3 -m http.server "$WHEEL_PORT" --bind 127.0.0.1 ) \
  >/tmp/trackg-wheelserver.log 2>&1 &
WHEEL_SRV_PID=$!
cleanup() { kill "$WHEEL_SRV_PID" 2>/dev/null || true; rm -f "$BUILT_ENV"; }
trap cleanup EXIT
sleep 2
curl -fsS "http://127.0.0.1:$WHEEL_PORT/" >/dev/null \
  || { echo "ERROR: wheel server not reachable on 127.0.0.1:$WHEEL_PORT"; exit 1; }

echo "=============================================="
echo "cotainr build"
echo "  env : $ENV_YML (port-substituted -> $BUILT_ENV)"
echo "  out : $SIF_OUT"
echo "  cotainr: $(command -v cotainr)"
echo "=============================================="

# --system rocm-6.2: the LUMI ROCm 6.2.4 base image (lumi-rocm-rocm-6.2.4.sif),
# which exactly matches our torch rocm6.2.4 wheel. NB: --system lumi-g is broken on
# this filesystem (its base-image symlink points into an unavailable laifs path:
# .../lumi-multitorch-mpich-...sif -> no such file); rocm-6.2 resolves and is the
# better userspace match anyway. The torch rocm wheel self-bundles the rest.
cotainr build "$SIF_OUT" \
  --system rocm-6.2 \
  --conda-env "$BUILT_ENV" \
  --accept-licenses \
  --log-to-file \
  -v

echo "=============================================="
echo "DONE: $SIF_OUT"
ls -lh "$SIF_OUT"
echo "Verify with: ./oellm/pipelines/container/verify_container_lumi.sh"
echo "=============================================="
