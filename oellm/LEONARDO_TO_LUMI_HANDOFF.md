# Handoff: synthif SFT pipeline — Leonardo → LUMI migration

**Date:** 2026-06-19
**Author of work:** abhash-er (abhash.kumar.jha@tue.ellis.eu)
**Reason for handoff:** Lost run rights on Leonardo; migrating the whole pipeline to LUMI (LUMI-G, AMD MI250X / ROCm).

---

## 1. Goal of the project

Port a multilingual SFT training pipeline (originally built for HoreKa A100) to an HPC
cluster, and run a set of **`synthif`** experiments:

- 5 data mixtures, each at **two scales** (500k and 1M samples)
- English ratios of **0 / 25 / 50 / 75 / 100 %**
- Base model: **Olmo-3-7B-Instruct-SFT**, trained with **OLMo-core** (`OLMo-sft.py`)

The 10 experiment configs already exist (untracked) in `oellm/configs/`:

```
synthif_500k_0en.yaml   synthif_1M_0en.yaml
synthif_500k_25en.yaml  synthif_1M_25en.yaml
synthif_500k_50en.yaml  synthif_1M_50en.yaml
synthif_500k_75en.yaml  synthif_1M_75en.yaml
synthif_500k_100en.yaml synthif_1M_100en.yaml
```

---

## 2. Repo / environment facts

- **Repo:** `open-instruct`, remote `ferreirafabio/open-instruct` (Fabio's fork), branch `main`.
- **OLMo-core checkout:** `…/fabio-open-instruct/OLMo-core` (sibling of `open-instruct`).
  - Has `src/scripts/train/sft/OLMo-sft.py` (training driver) and
    `src/examples/huggingface/convert_checkpoint_from_hf.py` (HF→OLMo-core converter).
  - This is the **newer, Beaker/NVIDIA-oriented** OLMo-core — only ONE `rocm` reference
    in `src/olmo_core/train/__init__.py`, and **no LUMI launch scripts bundled**.
    (Note: OLMo-2 was trained on LUMI, so `olmo_core` does support AMD — the port work is
    in the env + launch plumbing, not the training code.)
- **Env on Leonardo:** single `.venv` via `uv sync`; flash-attn was swapped for a
  glibc-2.28 prebuilt wheel (`flash_attn-2.8.3+cu128torch2.8-cp312-cp312-linux_x86_64.whl`,
  sitting in repo root). **This wheel + the cu128 torch are CUDA/NVIDIA — useless on LUMI.**
- **Leonardo SLURM:** account `oellm_prod2026`, GPU partition `boost_usr_prod`, 4× A100 64GB/node.

### Convert script interface (verified, unchanged across clusters)
`convert_checkpoint_from_hf.py` args: `--checkpoint-input-path`, `--model-arch`
(`olmo3_7b` is registered), `--tokenizer`, `--output-dir`, `--device`.

---

## 3. What was already DONE / verified on Leonardo

- ✅ OLMo-core source checkout present, with `OLMo-sft.py` + converter.
- ✅ Tokenizer in shared cache.
- ✅ Base HF checkpoint (Olmo-3-7B-Instruct-SFT) available in shared cache.
- ✅ Assembled data mixtures present.
- ✅ 10 `synthif_*.yaml` experiment configs written (untracked).
- ✅ Helper scripts written (untracked):
  - `oellm/pipelines/preprocessing/convert_base_to_olmocore_leonardo.sh`
  - `oellm/pipelines/tokenization/tokenize_trackG_leonardo.sh`
- ✅ Confirmed converter args + `olmo3_7b` arch registration match the scripts.

These data/CPU-side pieces (configs, tokenize, convert) are **mostly portable** to LUMI.

---

## 4. What was NOT done (remaining work)

1. **Tokenize** the 5 mixtures (script ready, never launched).
2. **Convert** base HF ckpt → OLMo-core format (script ready, never launched).
3. **Write the training launch script** (port of `train_multilingual_sft_horeka_a100.sh`) —
   never written.
4. **Launch** training.

---

## 5. The blocker that triggered the move

CUDA compatibility on Leonardo: the `.venv` torch is **cu128**, but Leonardo's host driver
caps at **CUDA 12.2**, and `module avail cuda` showed only **12.2 / 12.3 / 12.6** (no 12.8).
Whether cu128 torch runs on Leonardo GPUs via compat libs was unresolved — and then run
rights were lost. → Decision: **migrate to LUMI.**

---

## 6. LUMI migration — key facts (the CUDA → ROCm story)

LUMI-G is **AMD MI250X / ROCm 6.x**. The whole Python env is the wrong architecture and must
be rebuilt. Training code (`olmo_core`) is fine; env + launch layer get rewritten.

| Concern | Leonardo (old) | LUMI-G (new) |
|---|---|---|
| GPU / stack | NVIDIA A100, CUDA | AMD MI250X, ROCm 6.x |
| GPUs per node | 4 | **8 GCDs** (4× MI250X) → `--gpus-per-node=8` |
| Collectives | NCCL | **RCCL** + `aws-ofi-rccl` (Slingshot-11) |
| PyTorch | `uv sync` CUDA venv | LUMI **ROCm Singularity container** (`/appl/local/containers/sif-images/lumi-pytorch-rocm-*.sif`), `pip install -e` OLMo-core inside it |
| flash-attn | cu128 wheel | ROCm/CK build (often already in the container) |
| Account | `oellm_prod2026` | `project_465XXXXXX` |
| Partitions | `boost_usr_prod` | `standard-g`, `small-g`, `dev-g` (debug) |
| Filesystem | `/leonardo_work/…` | `/scratch/project_465…`, `/project/…`, `/flash/…` |
| Extra gotchas | — | per-node MIOpen cache dirs; GCD↔NUMA CPU-bind masks |

### Migration roadmap
1. **Access** — LUMI account + added to a `project_465…` GPU allocation; SSH key on MyAccessID.
2. **Move data** — tokenized datasets, base ckpt, tokenizer Leonardo→LUMI (Globus for tens of
   GB; rsync for small). Alternatively re-download HF base ckpt + re-convert on LUMI.
3. **Rebuild env for ROCm** — LUMI PyTorch ROCm container; install OLMo-core inside; ROCm flash-attn.
4. **Rewrite SLURM scripts** — new headers, `srun … singularity exec` wrapping, RCCL env
   (`NCCL_SOCKET_IFNAME`, CXI / aws-ofi-rccl), MIOpen cache, CPU-bind mask.
5. **Smoke-test on `dev-g`** (1 node, short) → scale to multi-node.

### What ports cleanly vs. gets rewritten
- **Portable:** the 10 `synthif_*.yaml` configs; tokenization script; HF→OLMo-core convert
  command (CPU/data work).
- **Rewritten:** the environment (CUDA venv → ROCm container) and the training launch script.

---

## 7. Open questions to resolve on LUMI (first thing next session)

- LUMI access status: account ready? added to a `project_465…` GPU allocation?
- Which LUMI project number / scratch path to use as the new project root?
- Transfer data over (Globus/rsync) vs. re-download base ckpt + re-tokenize on LUMI?
- Confirm the exact LUMI PyTorch ROCm container path/version available, and whether its
  flash-attn satisfies OLMo-core.

---

## 8. Quick reference — paths (Leonardo side, for the transfer)

- Project root: `/leonardo_work/OELLM_prod2026/users/ajha0001/fabio-open-instruct/open-instruct`
- OLMo-core: `/leonardo_work/OELLM_prod2026/users/ajha0001/fabio-open-instruct/OLMo-core`
- Configs: `oellm/configs/synthif_*.yaml`
- Tokenize script: `oellm/pipelines/tokenization/tokenize_trackG_leonardo.sh`
- Convert script: `oellm/pipelines/preprocessing/convert_base_to_olmocore_leonardo.sh`
