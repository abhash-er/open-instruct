# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "beaker-py>=1.32.2,<2.0",
#     "datasets>=4.0.0",
#     "numpy<2",
#     "ray[default]>=2.44.1",
#     "rich>=13.7.0",
#     "tqdm",
#     "transformers>=4.52.4",
#     "torch>=2.7.0,<2.8",
# ]
# ///

"""
This script converts SFT datasets to OLMoCore format. OLMoCore has a more efficient
implementation of the OLMo models (espeically for MoE), and so it can be preferable
to use it for training on next-token prediction tasks (e.g. SFT).

OLMoCore accepts data in numpy mmap format. One file is for the input tokens and one for the labels mask.

## Usage:
    ```bash
    python scripts/data/convert_sft_data_for_olmocore.py \
        --tokenizer_name_or_path allenai/OLMo-2-1124-7B \
        --dataset_mixer_list allenai/tulu-3-sft-olmo-2-mixture-0225 1.0 \
        --output_dir ./data/tulu-3-sft-olmo-2-mixture-0225-olmocore \
        --chat_template_name olmo
    ```

## Ai2 Internal Usage (requires `gantry>=3`):
    ```bash
    gantry run \
        --workspace ai2/jacobm \
        --budget ai2/oe-base \
        --priority normal \
        --cluster ai2/neptune --gpus 1 \
        --weka=oe-training-default:/weka/oe-training-default \
        --task-name convert-sft-data-for-olmocore --yes \
        --env-secret HF_TOKEN=HF_TOKEN \
        --install "echo 'do nothing'" \
        -- uv run --script scripts/data/convert_sft_data_for_olmocore.py \
            --dataset_mixer_list allenai/tulu-3-sft-olmo-2-mixture 1.0 \
            --tokenizer_name_or_path allenai/OLMo-2-1124-7B \
            --output_dir /weka/oe-training-default/ai2-llm/tylerr/data/sft/tulu-3-sft-olmo-2-mixture-0225-olmocore-test1 \
            --visualize True \
            --chat_template_name olmo \
            --max_seq_length 16384
    ```

    Dependencies for this script when run with `uv` are declared at the top of the file. `uv` will
    automatically install them *and not the project dependencies*.

    Add `--show-logs` to stream the logs to the terminal. By default `gantry run` will detach the job.

NOTE: allenai/OLMo-2-1124-7B tokenizer is the same as allenai/dolma2-tokenizer, but allenai/OLMo-2-1124-7B
has additional metadata required for this script.

Recommendations:
  * Set max_seq_length, and use the same length you use during SFT
"""

import gzip
import json
import os
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
from tqdm import tqdm

from open_instruct.dataset_transformation import (
    ATTENTION_MASK_KEY,
    DATASET_ORIGIN_KEY,
    INPUT_IDS_KEY,
    LABELS_KEY,
    TOKENIZED_SFT_DATASET_KEYS_WITH_SOURCE,
    TokenizerConfig,
    get_cached_dataset_tulu_with_statistics,
    remove_dataset_source_field,
    visualize_token,
)
from open_instruct.utils import ArgumentParserPlus, is_beaker_job

# Overridable; defaults next to this script. The original hardcoded HoreKa path
# (/work/dlclarge2/...) does not exist on other clusters and crashed the run.
DEBUG_LOG_PATH = os.environ.get(
    "OLMOCORE_DEBUG_LOG_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "olmocore_debug.log"),
)


def append_debug_log(
    session_id: str,
    run_id: str,
    hypothesis_id: str,
    location: str,
    message: str,
    data: Dict[str, Any],
) -> None:
    """Write a single NDJSON entry to the debug log."""
    log_entry = {
        "sessionId": session_id,
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    # Debug logging must never be fatal to the actual job.
    try:
        os.makedirs(os.path.dirname(DEBUG_LOG_PATH), exist_ok=True)
        with open(DEBUG_LOG_PATH, "a") as log_file:
            log_file.write(json.dumps(log_entry) + "\n")
    except OSError:
        pass


def save_checkpoint(output_dir: str, checkpoint_data: Dict[str, Any]) -> None:
    """Save lightweight checkpoint to disk atomically."""
    checkpoint_path = os.path.join(output_dir, "_checkpoint.json")
    tmp_path = checkpoint_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(checkpoint_data, f, indent=2)
    os.rename(tmp_path, checkpoint_path)  # Atomic on POSIX
    print(f"Checkpoint saved: {checkpoint_data['samples_processed']} samples, {checkpoint_data['chunks_written']} chunks")


def load_checkpoint(output_dir: str) -> Optional[Dict[str, Any]]:
    """Load checkpoint from disk if it exists."""
    checkpoint_path = os.path.join(output_dir, "_checkpoint.json")
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            return json.load(f)
    return None


def remove_checkpoint(output_dir: str) -> None:
    """Remove checkpoint file after successful completion."""
    checkpoint_path = os.path.join(output_dir, "_checkpoint.json")
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"Removed checkpoint file: {checkpoint_path}")


class PairedStreamingMemmapWriter:
    """Write token_ids and labels_mask to memmap files with aligned chunks.

    Both arrays are flushed together to ensure chunk boundaries match.
    This prevents OOM by keeping only a buffer in memory instead of all tokens.
    """

    def __init__(
        self,
        output_dir: str,
        token_dtype,
        max_tokens_per_chunk: int = 500_000_000,  # ~1GB for uint16
        start_chunk_idx: int = 0,
        start_total_written: int = 0,
    ):
        self.output_dir = output_dir
        self.token_dtype = token_dtype
        self.max_tokens_per_chunk = max_tokens_per_chunk
        self.token_buffer: List[int] = []
        self.labels_buffer: List[int] = []
        self.chunk_idx = start_chunk_idx
        self.total_written = start_total_written
        self.chunk_boundaries: List[Tuple[int, int]] = []

    def write(self, tokens: List[int], labels: List[int]) -> None:
        """Add tokens and labels to buffer, flush if threshold exceeded."""
        assert len(tokens) == len(labels), f"Token/label length mismatch: {len(tokens)} vs {len(labels)}"
        self.token_buffer.extend(tokens)
        self.labels_buffer.extend(labels)

        if len(self.token_buffer) >= self.max_tokens_per_chunk:
            self._flush_buffer()

    def _flush_buffer(self) -> None:
        """Write both buffers to disk as new memmap chunks."""
        if not self.token_buffer:
            return

        # Write token_ids
        token_filename = f"{self.output_dir}/token_ids_part_{self.chunk_idx:04d}.npy"
        token_arr = np.array(self.token_buffer, dtype=self.token_dtype)
        token_mmap = np.memmap(token_filename, mode="w+", dtype=self.token_dtype, shape=(len(token_arr),))
        token_mmap[:] = token_arr
        token_mmap.flush()
        del token_mmap

        # Write labels_mask (same chunk boundary)
        labels_filename = f"{self.output_dir}/labels_mask_part_{self.chunk_idx:04d}.npy"
        labels_arr = np.array(self.labels_buffer, dtype=np.bool_)
        labels_mmap = np.memmap(labels_filename, mode="w+", dtype=np.bool_, shape=(len(labels_arr),))
        labels_mmap[:] = labels_arr
        labels_mmap.flush()
        del labels_mmap

        chunk_start = self.total_written
        chunk_end = self.total_written + len(self.token_buffer)
        self.chunk_boundaries.append((chunk_start, chunk_end))

        token_size_mb = len(self.token_buffer) * np.dtype(self.token_dtype).itemsize / 1024**2
        labels_size_mb = len(self.labels_buffer) * np.dtype(np.bool_).itemsize / 1024**2
        print(f"Written chunk {self.chunk_idx:04d}: {len(self.token_buffer):,} tokens "
              f"(tokens: {token_size_mb:.1f} MB, labels: {labels_size_mb:.1f} MB)")

        self.total_written += len(self.token_buffer)
        self.token_buffer = []
        self.labels_buffer = []
        self.chunk_idx += 1

    def finalize(self) -> List[Tuple[int, int]]:
        """Flush remaining buffer and return chunk boundaries."""
        self._flush_buffer()
        return self.chunk_boundaries


@dataclass
class ConvertSFTDataArguments:
    """
    Arguments for converting SFT data to OLMoCore format.
    """

    """Output directory"""
    output_dir: str = field()

    """The name of the dataset to use (via the datasets library)."""
    dataset_name: Optional[str] = field(default=None)

    """A dictionary of datasets (local or HF) to sample from."""
    dataset_mixer: Optional[dict] = field(default=None)

    """A list of datasets (local or HF) to sample from."""
    dataset_mixer_list: List[str] = field(default_factory=lambda: ["allenai/tulu-3-sft-olmo-2-mixture-0225", "1.0"])

    """The dataset splits to use for training"""
    dataset_mixer_list_splits: List[str] = field(default_factory=lambda: ["train"])

    """The list of transform functions to apply to the dataset."""
    dataset_transform_fn: list[str] = field(
        default_factory=lambda: ["sft_tulu_tokenize_and_truncate_v1", "sft_tulu_filter_v1"]
    )

    """The columns to use for the dataset."""
    dataset_target_columns: List[str] = field(default_factory=lambda: TOKENIZED_SFT_DATASET_KEYS_WITH_SOURCE)

    """The mode to use for caching the dataset."""
    dataset_cache_mode: Literal["hf", "local"] = "local"

    """The directory to save the local dataset cache to."""
    dataset_local_cache_dir: str = "local_dataset_cache"

    """The hash of the dataset configuration."""
    dataset_config_hash: Optional[str] = None

    """Whether to skip the cache."""
    dataset_skip_cache: bool = False

    """Maximum sequence length. If not provided, no truncation will be performed."""
    max_seq_length: Optional[int] = field(default=None)

    """Number of examples to process for debugging. 0 means process all examples."""
    num_examples: int = field(default=0)

    """Visualize first token sequence"""
    visualize: bool = field(default=False)

    """Only write the tokenizer config to the output directory"""
    tokenizer_config_only: bool = field(default=False)

    """Maximum chunk size in MB for streaming writes (default 1024 = 1GB)"""
    max_chunk_size_mb: int = field(default=1024)

    """Shuffle seed for reproducible dataset ordering"""
    shuffle_seed: int = field(default=42)

    """Resume from checkpoint if available"""
    resume: bool = field(default=False)

    """Save checkpoint every N samples (0 = no checkpoints)"""
    checkpoint_interval: int = field(default=50000)


def main(args: ConvertSFTDataArguments, tc: TokenizerConfig):
    args.dataset_local_cache_dir = os.path.abspath(args.dataset_local_cache_dir)
    if is_beaker_job():
        beaker_cache_dir = "/weka/oe-adapt-default/allennlp/deletable_open_instruct_dataset_cache"
        if os.path.exists(beaker_cache_dir):
            args.dataset_local_cache_dir = beaker_cache_dir

    # region agent log
    append_debug_log(
        session_id="debug-session",
        run_id="run1",
        hypothesis_id="H1",
        location="convert_sft_data_for_olmocore.py:main_start",
        message="starting main",
        data={
            "dataset_mixer_list": args.dataset_mixer_list,
            "max_seq_length": args.max_seq_length,
        },
    )
    # endregion agent log

    print("Verify these values match the tokenizer config used in Olmo-core:")
    print(f"Tokenizer vocab_size: {tc.tokenizer.vocab_size}")
    print(f"Tokenizer bos_token_id: {tc.tokenizer.bos_token_id}")
    print(f"Tokenizer pad_token_id: {tc.tokenizer.pad_token_id}")
    print(f"Tokenizer eos_token_id: {tc.tokenizer.eos_token_id}")
    print(f"Tokenizer chat_template: {tc.tokenizer.chat_template}")

    output_dir = args.output_dir

    # Check if tokenization already completed
    stats_file = os.path.join(output_dir, "dataset_statistics.json")
    if os.path.exists(stats_file):
        print("=" * 50)
        print("TOKENIZATION ALREADY COMPLETE")
        print("=" * 50)
        print(f"Found completion marker: {stats_file}")
        print(f"Output directory: {output_dir}")
        print()
        print("To re-run tokenization, delete the output directory first:")
        print(f"  rm -rf {output_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)

    tokenizer_output_dir = os.path.join(output_dir, "tokenizer")
    os.makedirs(tokenizer_output_dir, exist_ok=True)
    print(f"Saving tokenizer to {tokenizer_output_dir}...")
    tc.tokenizer.save_pretrained(tokenizer_output_dir)

    # Check if chat_template.jinja exists and add it to tokenizer_config.json
    chat_template_path = os.path.join(tokenizer_output_dir, "chat_template.jinja")
    tokenizer_config_path = os.path.join(tokenizer_output_dir, "tokenizer_config.json")
    if os.path.exists(chat_template_path) and os.path.exists(tokenizer_config_path):
        with open(chat_template_path, "r") as f:
            chat_template_content = f.read()
        with open(tokenizer_config_path, "r") as f:
            tokenizer_config = json.load(f)
        if "chat_template" not in tokenizer_config:
            tokenizer_config["chat_template"] = chat_template_content
            with open(tokenizer_config_path, "w") as f:
                json.dump(tokenizer_config, f, indent=2)
            print(f"Added chat_template from {chat_template_path} to tokenizer_config.json")

    print("Tokenizer saved successfully!")

    if args.tokenizer_config_only:
        return

    # TODO: improve configurability of transform factory
    transform_functions_and_args = [
        ("sft_tulu_tokenize_and_truncate_v1", {"max_seq_length": args.max_seq_length}),
        ("sft_tulu_filter_v1", {}),  # remove examples that don't have any labels
    ]

    # region agent log
    append_debug_log(
        session_id="debug-session",
        run_id="run1",
        hypothesis_id="H2",
        location="convert_sft_data_for_olmocore.py:dataset_loading_start",
        message="starting dataset load",
        data={
            "dataset_mixer_list": args.dataset_mixer_list,
            "max_seq_length": args.max_seq_length,
        },
    )
    # endregion agent log

    train_dataset, dataset_statistics = get_cached_dataset_tulu_with_statistics(
        dataset_mixer_list=args.dataset_mixer_list,
        dataset_mixer_list_splits=args.dataset_mixer_list_splits,
        tc=tc,
        dataset_transform_fn=[func for func, _ in transform_functions_and_args],
        transform_fn_args=[args for _, args in transform_functions_and_args],
        target_columns=args.dataset_target_columns,
        dataset_cache_mode=args.dataset_cache_mode,
        dataset_config_hash=args.dataset_config_hash,
        dataset_local_cache_dir=args.dataset_local_cache_dir,
        dataset_skip_cache=args.dataset_skip_cache,
        drop_dataset_source=False,
        keep_dataset_in_memory=False,
    )

    # region agent log
    append_debug_log(
        session_id="debug-session",
        run_id="run1",
        hypothesis_id="H2",
        location="convert_sft_data_for_olmocore.py:dataset_loaded",
        message="dataset loaded",
        data={
            "dataset_length": len(train_dataset),
            "dataset_mixer_list": args.dataset_mixer_list,
        },
    )
    # endregion agent log

    train_dataset = train_dataset.shuffle(seed=args.shuffle_seed)
    print(f"Shuffled dataset with seed={args.shuffle_seed}")

    if args.visualize:
        print("Visualizing first example...")
        visualize_token(train_dataset[0][INPUT_IDS_KEY], tc.tokenizer)
        print("Labels:", train_dataset[0][LABELS_KEY])
        print("Attention mask:", train_dataset[0][ATTENTION_MASK_KEY])

    if args.num_examples > 0:
        print(f"Selecting {args.num_examples} examples for debugging")
        train_dataset = train_dataset.select(range(args.num_examples))

    total_samples = len(train_dataset)

    # Choose dtype based on vocab size - Olmo-core does the
    # same operation to infer the dtype of the token_ids array.
    vocab_size = tc.tokenizer.vocab_size
    token_dtype = None
    for dtype in (np.uint8, np.uint16, np.uint32, np.uint64):
        if (vocab_size - 1) <= np.iinfo(dtype).max:
            token_dtype = dtype
            print(f"Using dtype '{dtype}' for token_ids based on vocab size {vocab_size}")
            break
    if token_dtype is None:
        raise ValueError(f"Vocab size {vocab_size} is too big for any numpy integer dtype!")

    # Load checkpoint if resuming
    checkpoint = load_checkpoint(output_dir) if args.resume else None
    start_idx = 0
    start_chunk_idx = 0
    start_total_written = 0

    if checkpoint:
        start_idx = checkpoint["samples_processed"]
        start_chunk_idx = checkpoint["chunks_written"]
        start_total_written = checkpoint["total_tokens_written"]
        print(f"Resuming from checkpoint: {start_idx:,} samples processed, {start_chunk_idx} chunks written")

        if start_idx >= total_samples:
            print("All samples already processed. Nothing to do.")
            return

        # Efficiently skip already-processed samples
        train_dataset = train_dataset.select(range(start_idx, total_samples))
        print(f"Skipped to sample {start_idx:,}, processing remaining {len(train_dataset):,}")

    # Initialize paired streaming writer for incremental disk writes
    # Calculate max tokens per chunk based on token dtype size
    token_size = np.dtype(token_dtype).itemsize
    max_tokens_per_chunk = (args.max_chunk_size_mb * 1024**2) // token_size
    print(f"Streaming writes: max {max_tokens_per_chunk:,} tokens per chunk (~{args.max_chunk_size_mb} MB)")

    writer = PairedStreamingMemmapWriter(
        output_dir, token_dtype, max_tokens_per_chunk,
        start_chunk_idx=start_chunk_idx,
        start_total_written=start_total_written,
    )

    print("Collecting and streaming tokens from dataset...")
    sample: Mapping[str, Any]
    num_samples_skipped = 0
    # Load document_boundaries from checkpoint or start fresh
    document_boundaries = checkpoint.get("document_boundaries", []) if checkpoint else []
    # Convert from lists back to tuples if loaded from JSON
    document_boundaries = [tuple(b) for b in document_boundaries]
    if document_boundaries:
        print(f"Loaded {len(document_boundaries)} document boundaries from checkpoint")
    current_position = start_total_written  # Continue from checkpoint position

    # Running totals (since we no longer keep all tokens in memory)
    total_tokens = start_total_written
    total_trainable_tokens = checkpoint.get("total_trainable_tokens", 0) if checkpoint else 0

    # Track per-dataset statistics using dataset_source field
    per_dataset_counts = checkpoint.get("per_dataset_counts", {}) if checkpoint else {}
    per_dataset_tokens = checkpoint.get("per_dataset_tokens", {}) if checkpoint else {}
    per_dataset_trainable_tokens = checkpoint.get("per_dataset_trainable_tokens", {}) if checkpoint else {}
    per_dataset_filtered = checkpoint.get("per_dataset_filtered", {}) if checkpoint else {}
    num_samples_skipped = checkpoint.get("num_samples_skipped", 0) if checkpoint else 0

    for idx, sample in enumerate(
        tqdm(  # type: ignore
            train_dataset,
            desc="Collecting tokens",
            file=sys.stdout,
            bar_format="{l_bar}{bar}{r_bar}\n",  # better printing in beaker
            mininterval=10.0,
            initial=start_idx,  # Start progress bar at checkpoint position (global)
            total=total_samples,  # Always show total samples for consistent display
        )
    ):
        sample_length = len(sample[INPUT_IDS_KEY])
        sample_tokens = sample[INPUT_IDS_KEY]
        sample_labels = sample[LABELS_KEY]
        dataset_source = sample.get(DATASET_ORIGIN_KEY, "unknown")

        # Initialize counters for new datasets
        if dataset_source not in per_dataset_counts:
            per_dataset_counts[dataset_source] = 0
            per_dataset_tokens[dataset_source] = 0
            per_dataset_trainable_tokens[dataset_source] = 0
            per_dataset_filtered[dataset_source] = 0

        # Update per-dataset statistics
        per_dataset_counts[dataset_source] += 1
        per_dataset_tokens[dataset_source] += sample_length
        trainable_tokens_in_sample = sum(1 for label in sample_labels if label != -100)
        per_dataset_trainable_tokens[dataset_source] += trainable_tokens_in_sample

        # Stream tokens and labels to disk (writes chunk when buffer is full)
        writer.write(
            list(sample_tokens),
            [1 if label != -100 else 0 for label in sample_labels]
        )

        # Update running totals
        total_tokens += sample_length
        total_trainable_tokens += trainable_tokens_in_sample

        # Record document boundary (start, end)
        document_boundaries.append((current_position, current_position + sample_length))
        current_position += sample_length

        if all(label == -100 for label in sample_labels):
            num_samples_skipped += 1
            per_dataset_filtered[dataset_source] += 1

        # Assert that attention mask is all 1s
        assert all(mask == 1 for mask in sample[ATTENTION_MASK_KEY]), (
            f"Expected all attention mask values to be 1, but found: {sample[ATTENTION_MASK_KEY]}"
        )

        if idx % 500_000 == 0:
            # region agent log
            append_debug_log(
                session_id="debug-session",
                run_id="run1",
                hypothesis_id="H1",
                location="convert_sft_data_for_olmocore.py:collecting_tokens",
                message="collection progress",
                data={
                    "idx": idx,
                    "total_tokens": total_tokens,
                    "chunks_written": writer.chunk_idx,
                },
            )
            # endregion agent log

        # Save checkpoint periodically (idx is local after select, so add start_idx for global)
        global_samples_processed = start_idx + idx + 1
        if args.checkpoint_interval > 0 and global_samples_processed % args.checkpoint_interval == 0:
            save_checkpoint(output_dir, {
                "samples_processed": global_samples_processed,
                "chunks_written": writer.chunk_idx,
                "total_tokens_written": total_tokens,
                "total_trainable_tokens": total_trainable_tokens,
                "num_samples_skipped": num_samples_skipped,
                "per_dataset_counts": per_dataset_counts,
                "per_dataset_tokens": per_dataset_tokens,
                "per_dataset_trainable_tokens": per_dataset_trainable_tokens,
                "per_dataset_filtered": per_dataset_filtered,
                "document_boundaries": document_boundaries,
            })

    train_dataset = remove_dataset_source_field(train_dataset)

    # Finalize streaming writer (flush remaining buffer)
    print("Finalizing streaming writes...")
    chunk_boundaries = writer.finalize()

    # Verify consistency
    assert writer.total_written == total_tokens, (
        f"Token count mismatch: {writer.total_written} vs {total_tokens}"
    )

    total_instances = len(train_dataset)
    print(f"Total sequences: {total_instances}")
    print(f"Total tokens: {total_tokens}")
    print(f"Trainable tokens: {total_trainable_tokens}")
    print(f"Chunks written: {len(chunk_boundaries)}")
    print(f"Number of samples with no labels (skipped): {num_samples_skipped}")

    def write_metadata_for_chunks(base_filename, document_boundaries, chunk_boundaries):
        """Write metadata files for each chunk with document boundaries."""

        for chunk_idx, (chunk_start, chunk_end) in enumerate(chunk_boundaries):
            metadata_filename = f"{base_filename}_part_{chunk_idx:04d}.csv.gz"

            with gzip.open(metadata_filename, "wt") as f:
                # Find all documents that overlap with this chunk
                for doc_start, doc_end in document_boundaries:
                    # Check if document overlaps with chunk
                    if doc_end > chunk_start and doc_start < chunk_end:
                        # Adjust boundaries relative to chunk start
                        adjusted_start = max(0, doc_start - chunk_start)
                        adjusted_end = min(chunk_end - chunk_start, doc_end - chunk_start)

                        # Only write if there's actual content in this chunk
                        if adjusted_end > adjusted_start:
                            f.write(f"{adjusted_start},{adjusted_end}\n")

            print(f"Written metadata {metadata_filename}")

    # Write document boundary metadata for each chunk
    write_metadata_for_chunks(f"{output_dir}/token_ids", document_boundaries, chunk_boundaries)

    print("Data conversion completed successfully!")

    # Write dataset statistics
    write_dataset_statistics(
        output_dir=output_dir,
        dataset_statistics=dataset_statistics,
        total_instances=total_instances,
        total_tokens=total_tokens,
        total_trainable_tokens=total_trainable_tokens,
        num_samples_skipped=num_samples_skipped,
        tokenizer_name=tc.tokenizer_name_or_path,
        max_seq_length=args.max_seq_length,
        chat_template_name=tc.chat_template_name,
        per_dataset_counts=per_dataset_counts,
        per_dataset_tokens=per_dataset_tokens,
        per_dataset_trainable_tokens=per_dataset_trainable_tokens,
        per_dataset_filtered=per_dataset_filtered,
    )

    # Remove checkpoint after successful completion
    remove_checkpoint(output_dir)
    print("All processing completed successfully!")


def write_dataset_statistics(
    output_dir: str,
    dataset_statistics: Dict[str, Any],
    total_instances: int,
    total_tokens: int,
    total_trainable_tokens: int,
    num_samples_skipped: int,
    tokenizer_name: str,
    max_seq_length: Optional[int],
    chat_template_name: Optional[str],
    per_dataset_counts: Dict[str, int],
    per_dataset_tokens: Dict[str, int],
    per_dataset_trainable_tokens: Dict[str, int],
    per_dataset_filtered: Dict[str, int],
):
    """Write dataset statistics to both text and JSON files."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Merge pre-transformation stats with post-shuffle actual counts
    merged_stats = []
    pre_transform_stats = {stat["dataset_name"]: stat for stat in dataset_statistics.get("per_dataset_stats", [])}

    for dataset_name in per_dataset_counts:
        pre_stat = pre_transform_stats.get(dataset_name, {})
        merged_stat = {
            "dataset_name": dataset_name,
            "dataset_split": pre_stat.get("dataset_split", "unknown"),
            "initial_instances": pre_stat.get("initial_instances", "N/A"),
            "instances_after_transformation": pre_stat.get("final_instances", "N/A"),
            "instances_filtered_during_transformation": pre_stat.get("instances_filtered", "N/A"),
            "frac_or_num_samples": pre_stat.get("frac_or_num_samples"),
            # Upsampling information
            "original_dataset_size": pre_stat.get("original_dataset_size"),
            "is_upsampled": pre_stat.get("is_upsampled", False),
            "upsampling_factor": pre_stat.get("upsampling_factor", 1.0),
            # Post-shuffle actual statistics
            "final_instances_in_output": per_dataset_counts[dataset_name],
            "final_tokens_in_output": per_dataset_tokens[dataset_name],
            "final_trainable_tokens_in_output": per_dataset_trainable_tokens[dataset_name],
            "instances_filtered_after_tokenization": per_dataset_filtered[dataset_name],
            "avg_tokens_per_instance": per_dataset_tokens[dataset_name] / per_dataset_counts[dataset_name]
            if per_dataset_counts[dataset_name] > 0
            else 0,
            "percentage_of_total_tokens": (per_dataset_tokens[dataset_name] / total_tokens * 100)
            if total_tokens > 0
            else 0,
            "percentage_of_total_instances": (per_dataset_counts[dataset_name] / total_instances * 100)
            if total_instances > 0
            else 0,
        }
        merged_stats.append(merged_stat)

    # Prepare statistics data
    stats_data = {
        "timestamp": timestamp,
        "output_directory": output_dir,
        "configuration": {
            "tokenizer": tokenizer_name,
            "max_sequence_length": max_seq_length,
            "chat_template": chat_template_name,
        },
        "per_dataset_statistics": merged_stats,
        "overall_statistics": {
            "total_datasets": len(per_dataset_counts),
            "total_instances": total_instances,
            "total_tokens": total_tokens,
            "trainable_tokens": total_trainable_tokens,
            "trainable_percentage": (total_trainable_tokens / total_tokens * 100) if total_tokens > 0 else 0,
            "instances_filtered": num_samples_skipped,
            "average_sequence_length": total_tokens / total_instances if total_instances > 0 else 0,
        },
    }

    # Write JSON file
    json_path = os.path.join(output_dir, "dataset_statistics.json")
    with open(json_path, "w") as f:
        json.dump(stats_data, f, indent=2)
    print(f"Written dataset statistics to {json_path}")

    # Write human-readable text file
    text_path = os.path.join(output_dir, "dataset_statistics.txt")
    with open(text_path, "w") as f:
        f.write("Dataset Statistics Report\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {timestamp}\n")
        f.write(f"Output Directory: {output_dir}\n\n")

        f.write("Configuration:\n")
        f.write("-" * 40 + "\n")
        f.write(f"- Tokenizer: {tokenizer_name}\n")
        f.write(f"- Max Sequence Length: {max_seq_length}\n")
        f.write(f"- Chat Template: {chat_template_name}\n\n")

        f.write("Per-Dataset Statistics:\n")
        f.write("=" * 80 + "\n")

        for stat in stats_data["per_dataset_statistics"]:
            f.write(f"\nDataset: {stat['dataset_name']}\n")
            f.write(f"- Split: {stat['dataset_split']}\n")

            # Pre-transformation statistics
            f.write("\nPre-transformation:\n")
            f.write(f"  - Instances loaded: {stat.get('initial_instances', 'N/A')}\n")
            f.write(f"  - Instances after transformation: {stat.get('instances_after_transformation', 'N/A')}\n")
            f.write(
                f"  - Instances filtered during transformation: {stat.get('instances_filtered_during_transformation', 'N/A')}\n"
            )

            if stat.get("frac_or_num_samples") is not None:
                if isinstance(stat["frac_or_num_samples"], float):
                    f.write(f"  - Sampling fraction: {stat['frac_or_num_samples']}\n")
                else:
                    f.write(f"  - Sample count: {stat['frac_or_num_samples']}\n")

            # Show upsampling information if applicable
            if stat.get("is_upsampled", False):
                f.write(f"  - Original dataset size: {stat.get('original_dataset_size', 'N/A')}\n")
                f.write(f"  - Upsampling factor: {stat.get('upsampling_factor', 1.0):.2f}x\n")
                f.write(f"  - Upsampled to: {stat.get('instances_after_transformation', 'N/A')} instances\n")

            # Post-shuffle statistics (actual output)
            f.write("\nFinal output statistics (after shuffling):\n")
            f.write(f"  - Instances in output: {stat['final_instances_in_output']:,}\n")
            f.write(f"  - Total tokens: {stat['final_tokens_in_output']:,}\n")
            f.write(f"  - Trainable tokens: {stat['final_trainable_tokens_in_output']:,}\n")
            f.write(f"  - Instances with no labels: {stat['instances_filtered_after_tokenization']}\n")
            f.write(f"  - Average tokens per instance: {stat['avg_tokens_per_instance']:.1f}\n")
            f.write(f"  - Percentage of total tokens: {stat['percentage_of_total_tokens']:.1f}%\n")
            f.write(f"  - Percentage of total instances: {stat['percentage_of_total_instances']:.1f}%\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("Overall Statistics:\n")
        f.write("=" * 80 + "\n")
        f.write(f"- Total datasets: {stats_data['overall_statistics']['total_datasets']}\n")
        f.write(f"- Total instances: {stats_data['overall_statistics']['total_instances']:,}\n")
        f.write(f"- Total tokens: {stats_data['overall_statistics']['total_tokens']:,}\n")
        f.write(f"- Trainable tokens: {stats_data['overall_statistics']['trainable_tokens']:,} ")
        f.write(f"({stats_data['overall_statistics']['trainable_percentage']:.1f}%)\n")
        f.write(f"- Instances filtered out: {stats_data['overall_statistics']['instances_filtered']}\n")
        f.write(f"- Average sequence length: {stats_data['overall_statistics']['average_sequence_length']:.1f}\n")

    print(f"Written human-readable statistics to {text_path}")


if __name__ == "__main__":
    parser = ArgumentParserPlus((ConvertSFTDataArguments, TokenizerConfig))
    args, tc = parser.parse_args_into_dataclasses()
    main(args, tc)
