# LUMI Handbook — synthif SFT pipeline (OLMo-3-7B, OLMo-core)

A reproducible guide for running the `synthif` Track-G SFT experiments on
**LUMI-G (AMD MI250X / ROCm)**: set up the environment, then assemble → tokenize →
convert the base checkpoint → train. It documents the LUMI-specific environment
work and the source patches the pipeline needs, so you can recreate it from a
fresh checkout without copying anyone's workspace.

Companion doc: `LEONARDO_TO_LUMI_HANDOFF.md` explains *why* the pipeline moved to
LUMI; this one is *how to run it*.

---

## 1. What this pipeline produces

10 SFT runs of **Olmo-3-7B-Instruct-SFT**: 5 EU/English mixtures
(`G1-100en … G5-0en`) × 2 scales (`500k`, `1M`).

| Experiment | EU : English |
|---|---|
| G1-100en | 0 % EU / 100 % English |
| G2-75en  | 25 / 75 |
| G3-50en  | 50 / 50 |
| G4-25en  | 75 / 25 |
| G5-0en   | 100 / 0 |

---

## 2. Conventions — set these to your own locations

The provided scripts hardcode paths and a SLURM account at the top of each file;
edit those to your own before launching. Throughout this doc:

```bash
WORK=...                      # a project/scratch dir you can write to (a LUMI /scratch/<proj> path)
REPO=$WORK/open-instruct      # this repo
OLMO_CORE=$WORK/OLMo-core     # the OLMo-core fork checkout (branch: sft-slurm), sibling of REPO
CONTAINER=$WORK/containers/lumi-pytorch-rocm-6.2.1-python-3.12-pytorch-20240918-vllm-4075b35.sif
HF_HOME=$WORK/hf-cache        # shared HF cache (offline on compute nodes)
```

Pick your own SLURM **account** (a GPU allocation) and use the appropriate
partitions — that's site/project specific and not covered here.

> **`/scratch` vs `/pfs` gotcha:** on LUMI `/scratch/<proj>` is a host symlink to a
> `/pfs/...` path, and the container only bind-mounts the `/pfs` form. Always use
> the canonical `/pfs/...` path inside jobs/binds (the scripts normalize an
> inherited `/scratch/...` `HF_HOME` to `/pfs/...` for this reason).

**Partitions you'll want:** a CPU partition for tokenization (no GPU needed — see
§4), a short GPU partition for the base-checkpoint conversion and smoke tests, and
a multi-GPU partition (8 GCD/node) for full training.

---

## 3. The container

LUMI ships ROCm PyTorch Singularity containers under
`/appl/local/containers/sif-images/` (e.g. `lumi-pytorch-rocm-*.sif`). This
pipeline was built against the **Sept-2024** image
`lumi-pytorch-rocm-6.2.1-python-3.12-pytorch-20240918-vllm-...sif`. Copy one to
your `$WORK` and point `$CONTAINER` at it. Its conda env is activated with
`source /opt/miniconda3/bin/activate pytorch`.

> The exact image version matters — see §4 for why this stack drives the whole
> two-venv design.

---

## 4. Why two venvs / the version split (READ THIS FIRST)

That container is a **Sept-2024** build: `torch 2.6.0.dev`, `transformers 4.45.1`,
`vllm 0.6.3`, `tokenizers 0.20`. Too old for some Olmo-3 tooling, but upgrading it
globally breaks other parts. The resolution is **two separate venvs**, both layered
on the container via `python -m venv --system-site-packages`:

**`venv` — training + tokenization (container stack as-is).**
- Olmo-3's tokenizer loads fine on transformers 4.45.1 (vocab 100278); open-instruct
  injects its own `olmo` chat template, so the tokenizer's empty `chat_template`
  is irrelevant.
- Only `AutoConfig.from_pretrained` chokes on the `olmo3` arch in 4.45.1 → patched
  to fall back gracefully (§5 patch list).
- vllm 0.6.3 probes the GPU at import and is too old → its import is **neutralized**
  in `open_instruct/utils.py`, so tokenization needs **no GPU** and runs on a CPU
  partition.

**`venv-convert` — HF→OLMo-core conversion only.**
- The converter hard-requires `transformers ≥ 4.57` (for `Olmo3Config`), which needs
  `torch ≥ 2.7`. That can't live in the training venv (would break training's
  torch 2.6), so it gets its own venv. Training's `OLMo-sft.py` imports neither
  `transformers` nor `nn.hf`, so the two stacks never collide.

---

## 5. Environment setup

### 5a. Build the training venv (`venv`)
Layer open-instruct's deps on the container's conda env. First create an
"excludes" list that keeps container-provided packages pinned to the container
copies instead of letting `uv` resolve them:

```bash
cat > "$WORK/excludes-container.txt" <<'EOF'
torch
pytorch-triton-rocm
triton
vllm
flash-attn
transformers
accelerate
peft
tokenizers
safetensors
huggingface-hub
EOF
```
Then build the venv:
```bash
singularity exec --bind "$WORK:$WORK" "$CONTAINER" bash -c '
  set -euo pipefail
  source /opt/miniconda3/bin/activate pytorch
  python3 -m venv --system-site-packages "'"$WORK"'/venv"
  source "'"$WORK"'/venv/bin/activate"
  uv pip install --excludes "'"$WORK"'/excludes-container.txt" -e "'"$REPO"'"
'
```

If you hit `ModuleNotFoundError: beaker`, the excludes-based build missed an
open-instruct import dep — install it into the venv: `uv pip install 'beaker-py>=2.5.0'`.

A convenience shell function to drop into the container with `venv` active:
```bash
olmo_env() {
  local sif="$CONTAINER" venv="$WORK/venv"
  singularity exec --bind "$WORK:$WORK" "$sif" bash -c "
    source /opt/miniconda3/bin/activate pytorch
    source '$venv/bin/activate'
    exec bash"
}
```

### 5b. Build the conversion venv (`venv-convert`) — only for §7b
Isolated from `venv`, with the newer stack the converter needs:
```bash
singularity exec --bind "$WORK:$WORK" "$CONTAINER" bash -c '
  set -euo pipefail
  source /opt/miniconda3/bin/activate pytorch
  python3 -m venv --system-site-packages "'"$WORK"'/venv-convert"
  source "'"$WORK"'/venv-convert/bin/activate"
  uv pip install --no-deps --index-url https://download.pytorch.org/whl/rocm6.2.4 "torch==2.7.1+rocm6.2.4"
  uv pip install "transformers==4.57.*"
  uv pip install --no-deps --index-url https://download.pytorch.org/whl/rocm6.2.4 "torchvision==0.22.1+rocm6.2.4"
  uv pip install "cached-path>=1.7.2" omegaconf importlib_resources bettermap rich
'
```
> Build on a **login node** — compute nodes have no internet. torch is a ~3.5 GB
> download. `torchvision` must match the torch version (0.22.1 ↔ 2.7.1) or you get
> `operator torchvision::nms does not exist` during conversion.

**Why these specific dependency overrides:**
- `--system-site-packages` so the venv inherits olmo_core's other runtime deps from
  the container's conda env; only the packages below are overridden on top.
- `torch` and `torchvision` use `--no-deps` + the ROCm wheel index so pip installs the
  exact ROCm builds **without** dragging in a conflicting CUDA torch or other deps.
- `transformers==4.57.*` is installed **with** deps, so it transitively upgrades
  `tokenizers` (→0.22), `huggingface-hub` (→0.36), `safetensors` (→0.8) and `numpy`
  (→2.x) in this venv. That's fine here — it's why the converter is **isolated**:
  these upgrades would break the training/tokenization `venv` (container torch 2.6 +
  transformers 4.45), so they must not be applied there.
- `cached-path omegaconf importlib_resources bettermap rich` are olmo_core imports not
  present in the container conda env; install them explicitly or `import olmo_core`
  fails (e.g. `ModuleNotFoundError: No module named 'cached_path'`).
- Pin transformers to the **4.57.x** series specifically: it has `Olmo3Config` (needed)
  and the HF per-layer-rope fix (PR #45945), while newer majors (e.g. 5.x) pull a
  much heavier upgrade set (numpy 2.5 / hub 1.x) than the container torch tolerates.

For the **training `venv`** the only non-container override is `beaker-py` (see 5a);
everything else is deliberately left at the container versions via the excludes list.

### 5c. Get the model + data into the offline cache
Compute nodes are offline, so pre-populate on a login node and export
`HF_HOME=$HF_HOME HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` everywhere:
- **Base model** `allenai/Olmo-3-7B-Instruct-SFT` (HF download → lands in `$HF_HOME/hub/...`).
- **Assembled mixtures** `synthif-<SCALE>-<EN_TAG>.parquet` (§7a).

---

## 6. Source patches the pipeline needs

Jobs run `python` against the live working tree, so editing the files is enough
(no reinstall). **Commit these to your forks** so they survive a fresh checkout.

**`open-instruct`:**

| File | Change | Why |
|---|---|---|
| `open_instruct/utils.py` | make `import vllm` optional (`try/except`); make the `from_vllm_config` annotation lazy | vllm 0.6.3 probes the GPU at import and lacks `VllmConfig`; lets tokenization run GPU-free on a CPU node |
| `open_instruct/dataset_transformation.py` | guard `AutoConfig.from_pretrained` in `get_tokenizer_tulu_v2_2` (fall back to a name-based `model_type`); make the debug-log path env-overridable + best-effort | transformers 4.45.1 doesn't recognize `olmo3`; a hardcoded debug-log path otherwise crashes |
| `scripts/data/convert_sft_data_for_olmocore.py` | make the debug-log path env-overridable + non-fatal | same hardcoded-path issue |

**`OLMo-core` fork (`sft-slurm`):**

| File | Change | Why |
|---|---|---|
| `src/olmo_core/nn/transformer/config.py` | `olmo3_7B` calls `with_rope_scaling(YaRNRoPEScalingConfig(), full_attn_layers_only=True)` (+ import) | The released Olmo-3 checkpoint uses **YaRN** rope (factor 8, ctx 8192→65536) on full-attention layers only; the arch was plain RoPE. Without this the converted checkpoint is wrong (validation diverges ~97%). |
| `src/examples/huggingface/convert_checkpoint_from_hf.py` | add an env-gated `OLMOCORE_VALIDATE_FP32` path (+ `os` import) | at bf16, matmul noise alone exceeds the 1e-4 validation tolerance even for a *correct* conversion; fp32 validates cleanly |

> `transformers 4.57.6` already includes HF PR #45945 (per-layer rope: scaled on
> full-attention layers, default on sliding), so the olmo_core YaRN change above is
> the **only** code change needed for a correct conversion.

---

## 7. Running the pipeline

The provided scripts already set HF-offline env, `HF_HOME` normalization,
`DS_ACCELERATOR=cpu` (stops deepspeed's accelerator auto-probe from hanging in
`amdsmi_init`), `ROCR_VISIBLE_DEVICES → HIP_VISIBLE_DEVICES` translation (ray
rejects the former on LUMI), and `PYTHONUNBUFFERED=1`. Edit the path/account
variables at the top of each script first.

### 7a. (If needed) assemble the mixtures
The `synthif-<SCALE>-<EN_TAG>.parquet` files are the tokenizer inputs. To rebuild:
```bash
oellm/assemble_1M.sh         # 1M configs; adapt for 500k
```

### 7b. Convert the base HF checkpoint → OLMo-core (one-shot, GPU)
Produces the shared base used by all 10 runs. Uses `venv-convert`, the **local HF
snapshot path** (passing the repo id triggers an offline `repo_exists()` failure),
and fp32 validation.
```bash
sbatch oellm/pipelines/preprocessing/convert_base_to_olmocore_lumi.sh
# → checkpoints/base/Olmo-3-7B-Instruct-SFT-olmocore/{config.json, model_and_optim/}
```
Expect `Validation completed successfully` in the log (a few minutes on one GCD).

### 7c. Tokenize the mixtures (CPU, array job)
```bash
sbatch --array=0-4 oellm/pipelines/tokenization/tokenize_trackG_lumi.sh           # 500k (default)
SCALE=1M sbatch --array=0-4 oellm/pipelines/tokenization/tokenize_trackG_lumi.sh  # 1M
# → data/datasets_multilingual_sft/tokenized/<SCALE>/<EXPERIMENT>/
#     token_ids_part_0000.npy, labels_mask_part_0000.npy, dataset_statistics.*, tokenizer/
```
Tokenizing 500k–1M samples at `max_seq_length=32768` takes **hours** of CPU. The
log can look frozen at `TorchCheckpointEngine Initialized` — that's buffered stdout,
not a hang; confirm progress with `sstat -j <jobid> --format=AveCPU,MaxRSS`.

### 7d. Smoke-test training (5 steps, 1 node)
```bash
EXPERIMENT=G1-100en sbatch oellm/pipelines/training/test_train_5step_lumi.sh
```

### 7e. Full training (multi-node)
```bash
EXPERIMENT=G1-100en SCALE=500k sbatch oellm/pipelines/training/train_synthif_sft_lumi.sh
# 1 srun task per GCD (8/node); olmo_core builds its own RCCL process group.
# Requires the tokenized dataset (§7c) and the base ckpt (§7b).
```

**Fresh-run order:** `5a/5b build venvs → 7b convert + 7c tokenize (parallel) →
7d smoke test → 7e full train`.

---

## 8. Troubleshooting — errors seen during the port and their fixes

| Symptom | Cause | Fix |
|---|---|---|
| Hang in `amdsmi_init` at import | deepspeed auto-probes the accelerator | `export DS_ACCELERATOR=cpu` |
| `Please use HIP_VISIBLE_DEVICES instead of ROCR_VISIBLE_DEVICES` | ray rejects LUMI's env var | translate `ROCR_→HIP_VISIBLE_DEVICES` |
| `FileNotFoundError` / invisible `HF_HOME` under `/scratch/…` | container only binds `/pfs` | use/normalize the canonical `/pfs/...` path |
| crash on a hardcoded `.../.cursor/debug.log` | debug-log path baked in | patched env-overridable + non-fatal |
| `vllm.config has no attribute VllmConfig` / `No HIP GPUs available` at import | container vllm 0.6.3 too old & GPU-probing | vllm import neutralized → tokenize on CPU |
| `olmo3 architecture not recognized` (AutoConfig) | transformers 4.45.1 | guarded AutoConfig fallback |
| `ModuleNotFoundError: beaker` | excludes-based venv build missed it | `uv pip install 'beaker-py>=2.5.0'` into the venv |
| `cannot import name 'Olmo2Config'` (conversion) | transformers 4.45.1 lacks it | use `venv-convert` (transformers 4.57) |
| `OfflineModeIsEnabled: Cannot reach huggingface.co` (conversion) | olmo_core `repo_exists()` on a repo id | pass the **local snapshot dir**, not the repo id |
| `operator torchvision::nms does not exist` (conversion) | torchvision built for a different torch | install matching `torchvision==0.22.1+rocm6.2.4` |
| conversion validation: ~97 % logit mismatch | `olmo3_7B` arch lacked YaRN rope | add YaRN on full-attention layers (olmo_core patch) |
| conversion validation: ~85 % mismatch, max abs diff ~0.2 | bf16 matmul noise vs 1e-4 tolerance (not a bug) | `OLMOCORE_VALIDATE_FP32=1` → validates clean |
| tokenize log frozen for hours | block-buffered stdout | `PYTHONUNBUFFERED=1`; check CPU with `sstat` |
