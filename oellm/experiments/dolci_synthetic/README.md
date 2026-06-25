# Track G — dolci-synthetic SFT

Fine-tunes the **OLMo-3-7B-Instruct-SFT** base on the **synthif** mixtures:
synthetic instruction-following data (EU languages) blended with **Dolci English
replay**, swept across the English ratio and dataset scale.

This is the LUMI-G (AMD MI250X / ROCm) port of the Leonardo pipeline.

## Experiment matrix (10 runs)

| Mixture   | English | EU (11 langs, equal split) |
|-----------|--------:|----------------------------|
| G1-100en  |  100%   |   0%                       |
| G2-75en   |   75%   |  25%                       |
| G3-50en   |   50%   |  50%                       |
| G4-25en   |   25%   |  75%                       |
| G5-0en    |    0%   | 100%                       |

× two scales: **500k** and **1M** samples (max_seq_len 32768, OLMo chat template).
EU languages: es, fr, de, it, pt, pl, nl, cs, ro, el, uk. Sources: `synthgen-if`
(synthetic IF) + `dolci-replay` (English replay). Per-mixture seed 42.

Reference stats (G1): 500k → 365.7M tokens (~731 tok/seq, 48.9% trainable);
1M → 730.0M tokens (~730 tok/seq). Token count shrinks monotonically G1→G5 as
EU text packs slightly differently.

## Pipeline (end to end)

The generic, reusable mechanics live under `oellm/pipelines/`; this experiment
dir only adds orchestration + the results writeup.

1. **Assemble mixtures** (upstream of this repo): produces
   `gen_outputs/.../assembled/synthif-<scale>-<en>.parquet`
   from the configs in `oellm/configs/synthif_{500k,1M}_*.yaml`
   (see `oellm/assemble_1M.sh`).

2. **Convert base checkpoint** HF → OLMo-core (one-shot, shared by all runs):
   ```bash
   sbatch oellm/pipelines/preprocessing/convert_base_to_olmocore_lumi.sh
   # -> checkpoints/base/Olmo-3-7B-Instruct-SFT-olmocore
   ```

3. **Tokenize** the mixtures into OLMo-core numpy layout (small-g, 1 GPU; the
   import chain wedges on a GPU-less node, and `TOKENIZERS_PARALLELISM=false`
   avoids the fork-after-threadpool deadlock):
   ```bash
   sbatch --array=0-4 oellm/pipelines/tokenization/tokenize_trackG_lumi.sh            # 500k
   SCALE=1M sbatch --array=0-4 oellm/pipelines/tokenization/tokenize_trackG_lumi.sh   # 1M
   # -> data/datasets_multilingual_sft/tokenized/<scale>/<EXP>/
   ```

4. **Train** all 10 SFT runs (smoke-test first on dev-g):
   ```bash
   # 5-step smoke test (1 node × 8 GCD)
   EXPERIMENT=G1-100en sbatch oellm/pipelines/training/test_train_5step_lumi.sh

   # full sweep (this experiment's orchestrator; handles scale-aware run names
   # and checkpoint dirs so 500k/1M don't collide):
   ./oellm/experiments/dolci_synthetic/scripts/launch_all_train.sh
   # subsets:
   EXPERIMENTS="G1-100en G5-0en" SCALES="1M" ./...launch_all_train.sh
   # -> checkpoints/synthif/synthif-<scale>-<EXP>/stepN
   ```

5. **Convert trained checkpoints** OLMo-core → HF (for eval / sharing):
   ```bash
   EXPERIMENT=G1-100en SCALE=500k \
     sbatch oellm/experiments/dolci_synthetic/scripts/convert_trained_to_hf_lumi.sh
   # -> checkpoints/synthif/synthif-<scale>-<EXP>/stepN-hf
   ```

## Scripts in this dir

- `scripts/launch_all_train.sh` — submit the full (or a subset of the) sweep.
  Owns scale-aware `RUN_NAME` / `OLMO_SAVE_FOLDER` so the two scales of a mixture
  don't overwrite each other (the generic train script's default name omits scale).
  `DRY_RUN=1` prints the sbatch lines without submitting.
- `scripts/convert_trained_to_hf_lumi.sh` — OLMo-core → HF for a trained run;
  auto-resolves the latest `step*` from `EXPERIMENT`+`SCALE`, or takes `CKPT_DIR`.

## Notes

- Account: filesystem under `project_462001516`, jobs charged to `project_465002530`.
- Container: `lumi-pytorch-rocm-6.2.1-...sif` with two layered venvs — `venv`
  (tokenization/training, transformers 4.45) and `venv-convert` (conversion,
  transformers 4.57 + torch 2.7.1+rocm6.2.4).
- Logs: `oellm/pipelines/*/logs/` (pipeline steps) and
  `oellm/experiments/dolci_synthetic/logs/` (conversion here). Both gitignored.
