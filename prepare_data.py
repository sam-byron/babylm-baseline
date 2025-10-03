import glob
import os
import re
# Disable the tokenizers' parallelism to avoid deadlocks when using multiprocessing
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import argparse
import gc
import random
# from sre_parse import Tokenizer
# from datasets import load_dataset, Dataset, concatenate_datasets, load_from_disk
from functools import partial
# Legacy concatenation path retained for reference, but we now implement direct block writer
from multiprocessing import Pool
import torch
import multiprocessing as mp
from itertools import islice
from transformers import AutoTokenizer
# from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel

from transformers.tokenization_utils_base import BatchEncoding

from tokenizer import Tokenizer
from sentence_packing import pack_sentences

SPEAKER_PATTERN = re.compile(r"^([A-Z][a-zA-Z]{1,20})(:| —|-)")

def load_first_names(path:"str"):
    names=set()
    try:
        with open(path,'r') as f:
            for line in f:
                line=line.strip()
                if not line or line.startswith('#'): continue
                names.add(line.split()[0])
    except Exception as e:
        print(f"[SpeakerTag] Warning: could not load first names list {path}: {e}")
    return names

# limit PyTorch to 1 thread per process as well
# torch.set_num_threads(1)

# Add safe globals for torch serialization
torch.serialization.add_safe_globals([BatchEncoding])

# Define the collate function at the top level to make it picklable by multiprocessing workers
def identity_collate_fn(examples):
    """An identity collate function that simply returns the batch as a list."""
    return examples
    
def chunked(iterable, size):
    it = iter(iterable)
    while True:
        try:
            # Attempt to get the next item from the iterator
            batch = list(islice(it, size))
            yield batch
        except StopIteration:
            # If StopIteration is raised, it means the iterator is exhausted
            break

class BertDataset(Dataset):
    def __init__(self, data_dir, tokenizer, prepacked_segments=None):
        self.tokenizer = tokenizer

        self.n_special_tokens = 6

        self.mask_index = self.tokenizer.token_to_id("[MASK]")
        self.cls_index = self.tokenizer.token_to_id("[CLS]")
        self.sep_index = self.tokenizer.token_to_id("[SEP]")
        self.pad_index = self.tokenizer.token_to_id("[PAD]")

        """Load all .md files as complete file contents, not line by line"""
        if prepacked_segments is not None:
            self.segments = prepacked_segments
        else:
            pattern = os.path.join(data_dir, "**/*.md")
            md_files = glob.glob(pattern, recursive=True)
            print(f"Found {len(md_files)} .md files")

            self.segments = []
            for file_path in md_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for i, segment in enumerate(f):
                            segment = segment.strip()
                            if segment:  # Only add non-empty content
                                # prepend [CLS] and append [SEP] to each segment
                                segment = f"[CLS] {segment} [SEP]"
                                self.segments.append(segment)
                except Exception as e:
                    print(f"Warning: Could not read {file_path}: {e}")

    def __len__(self):
        return len(self.segments)
    
    def shuffle(self):
        """Shuffle the dataset segments in place."""
        random.shuffle(self.segments)

    def __getitem__(self, index):
        # Handle both single index and list of indices
        if isinstance(index, list):
            # Return list of items for batch processing
            return [self._get_single_item(i) for i in index]
        else:
            # Return single item
            return self._get_single_item(index)
    
    def _get_single_item(self, index):
        """Get a single item by index"""
        segment = self.segments[index]
        # append [CLS] and [SEP] string to segment
        # formatted_segment = f"[CLS] {segment} [SEP]"
        
        # return formatted_segment
        return {"text": segment}
    
def load_md_files_to_dataset(data_dir, tokenizer):
    """Load all .md files as complete file contents, not line by line"""
    pattern = os.path.join(data_dir, "**/*.md")
    md_files = glob.glob(pattern, recursive=True)
    print(f"Found {len(md_files)} .md files")
    
    texts = []
    for file_path in md_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:  # Only add non-empty content
                    texts.append(content)
        except Exception as e:
            print(f"Warning: Could not read {file_path}: {e}")
    
    print(f"Loaded {len(texts)} text segments (complete files)")
    return Dataset.from_dict({"text": texts})

def iter_raw_lines(data_dir):
    pattern = os.path.join(data_dir, "**/*.md")
    for file_path in glob.glob(pattern, recursive=True):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line=line.strip()
                    if line:
                        yield line
        except Exception as e:
            print(f"Warning: Could not read {file_path}: {e}")

def prepare_data(config, tokenizer, cache_path, sentence_pack=False):

    # Ensure the cache folder exists
    if not os.path.exists(cache_path):
        os.makedirs(cache_path)
        print(f"Created cache folder: {cache_path}")

    # Check if chunk files exist
    # chunk_paths = sorted(glob.glob(os.path.join(cache_path, "chunk*.pt")))

    chunk_size = config["chunk_size"]
    block_size = config.get("block_size", config.get("max_position_embeddings", 128))

    # Load complete file contents, not line by line
    data_dir = "./data/pretrain/bnc"
    # ALWAYS ON: sentence packing + document & speaker tagging (fresh start design)
    print("[SentencePack] Collecting raw lines for packing (always-on)...")
    raw_lines = []
    packed_results = []
    names = load_first_names(("./first-names.txt"))  # currently unused but retained for future inline speaker tagging
    debug_max = config.get("debug_max_raw_lines")
    line_counter = 0
    doc_counter = 0
    pattern = os.path.join(data_dir, "**/*.md")
    md_files = sorted(glob.glob(pattern, recursive=True))
    escape_literal_markers = config.get("escape_literal_markers", False)
    literal_markers = config.get("literal_markers", ["[DOC]", "[EOD]", "[SPK]"])
    if escape_literal_markers:
        print(f"[LiteralEscape] Enabled; will escape markers inside raw text lines: {literal_markers}")
    for file_idx, file_path in enumerate(md_files):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                file_lines = []
                for raw in f:
                    raw = raw.strip()
                    if not raw:
                        continue
                    if escape_literal_markers:
                        # Escape occurrences of structural markers so tokenizer does not map them to special IDs
                        # Strategy: prefix a backslash to '[' to defeat exact-match special token lookup.
                        for m in literal_markers:
                            if m in raw:
                                raw = raw.replace(m, '\\' + m)
                    file_lines.append(raw)
                    line_counter += 1
                    if debug_max and line_counter >= debug_max:
                        break
            if file_lines:
                doc_counter += 1
                raw_lines.append('[DOC]')
                raw_lines.extend(file_lines)
                raw_lines.append('[EOD]')
        except Exception as e:
            print(f"[SentencePack] Warning: could not read {file_path}: {e}")
        if debug_max and line_counter >= debug_max:
            break
    print(f"[SentencePack] Documents wrapped: {doc_counter}")
    print(f"[SentencePack] Raw line count (with DOC/EOD markers): {len(raw_lines)} (payload lines + markers)")
    # We previously packed up to full positional capacity (e.g. 512) and then split down to smaller blocks.
    # New requirement: pack directly to block_size granularity to avoid post-hoc splitting.
    # Allow explicit override via `packing_max_len` but default to block_size.
    # Config keys (new):
    #  - packing_max_len: maximum tokens per packed segment before emission; default to block_size to avoid post-splitting.
    #  - blocks_per_file: number of fixed-size blocks to aggregate into a single chunk file (I/O efficiency).
    #  - oversize_policy: handling strategy for rare segments > block_size even after packing (split|truncate|skip|raise). Default: split.
    packing_max_len = config.get("packing_max_len", block_size)
    max_len = packing_max_len
    packed_result = pack_sentences(
        raw_lines,
        tokenizer,
        max_len,
        add_doc_and_speaker=True,
        collect_stats=True,
        drop_short_trailing=True,
    )
    # When collect_stats True we now get (segments, stats)
    if isinstance(packed_result, tuple):
        packed_segments, stats = packed_result
    else:
        packed_segments = packed_result
        stats = None
    print(f"[SentencePack] Produced {len(packed_segments)} packed segments")
    # If stats present print them with derived rates
    if stats:
        packs = max(stats.get('packs', 0), 1)
        terminal_rate = stats.get('terminal_end', 0) / packs
        nonterminal_rate = stats.get('nonterminal_end', 0) / packs
        borrow_rate = stats.get('borrow_moves', 0) / packs
        dangling_rate = stats.get('dangling_carried', 0) / packs
        short_drop_rate = stats.get('short_trailing_dropped', 0) / packs
        print("[SentencePack][Stats] Raw:")
        for k,v in stats.items():
            print(f"  - {k}: {v}")
        print("[SentencePack][Stats] Derived rates:")
        print(f"  - terminal_end_rate: {terminal_rate:.4f}")
        print(f"  - nonterminal_end_rate: {nonterminal_rate:.4f}")
        print(f"  - borrow_move_rate: {borrow_rate:.4f}")
        print(f"  - dangling_carried_rate: {dangling_rate:.4f}")
        print(f"  - short_trailing_dropped_rate: {short_drop_rate:.4f}")
    # Instead of batching string segments then concatenating, directly tokenize and stream out fixed-size blocks
    ds = BertDataset(data_dir, tokenizer, prepacked_segments=packed_segments)

    # Shuffling strategy: default = document-level (preserve intra-doc order)
    shuffle_level = config.get("shuffle_level", "document")  # document|segment|none
    print(f"[Shuffle] Strategy: {shuffle_level}")
    if shuffle_level == "segment":
        ds.shuffle()
    elif shuffle_level == "document":
        # Group consecutive segments between [DOC] and [EOD]
        grouped = []
        current = []
        for seg in ds.segments:
            txt = seg
            # detect DOC start
            if txt.startswith('[DOC]') and current:
                # edge case: stray new doc start without prior EOD
                grouped.append(current)
                current = []
            current.append(seg)
            if txt.rstrip().endswith('[EOD]'):
                grouped.append(current)
                current = []
        if current:
            grouped.append(current)
        random.shuffle(grouped)
        # flatten preserving internal order
        ds.segments = [s for g in grouped for s in g]
        print(f"[Shuffle] Documents grouped: {len(grouped)}")
    else:
        print("[Shuffle] No shuffling applied")

    print("First 5 samples:")
    for i in range(min(5, len(ds))):
        sample = ds[i]
        # print(f"Sample {i}: {sample['text']}...")
        print(f"Sample {i}: {sample}")
    # Print number of samples in the dataset
    print(f"Number of samples in dataset: {len(ds)}")

    # Print number of words in the dataset
    total_words = sum(len(ds[i]["text"].split()) for i in range(len(ds)))
    # total_words = sum(len(ds[i].split()) for i in range(len(ds)))
    print(f"Total words in dataset: {total_words}")
    # return

    # wrap the HuggingFace streaming IterableDataset in a PyTorch DataLoader
    # to parallelize I/O with num_workers > 1
    from torch.utils.data import DataLoader
    # ================= Boundary-aware block emission (Option 1: min_fill_ratio=0.70 with carry-forward) =================
    print(f"[BlockWriter] Buffered emission (block_size={block_size}, packing_max_len={packing_max_len})")
    pad_id = tokenizer.token_to_id("[PAD]") or 0

    # Buffer configuration
    blocks_per_file = config.get("blocks_per_file", 1000)  # each output .pt will contain this many blocks (except final tail)
    # Stats
    total_blocks = 0
    padded_blocks = 0
    total_tokens_before_padding = 0
    oversize_segments = 0  # should usually be 0 now that packing_max_len==block_size
    oversize_truncated_segments = 0
    oversize_split_segments = 0

    # In-memory buffers
    buffer_blocks: list[list[int]] = []
    buffer_block_metas: list[dict] = []
    file_index = 0

    def flush_file(final=False):
        nonlocal buffer_blocks, buffer_block_metas, file_index
        if not buffer_blocks:
            return
        tensor = torch.tensor(buffer_blocks, dtype=torch.long)
        out_path = os.path.join(cache_path, f"chunk{file_index}.pt")
        torch.save(tensor, out_path)
        meta = {
            "file_index": file_index,
            "num_blocks": len(buffer_blocks),
            "block_size": block_size,
            "blocks": buffer_block_metas,
            "final_file": final,
        }
        meta_path = os.path.join(cache_path, f"chunk{file_index}.meta.json")
        try:
            with open(meta_path, 'w', encoding='utf-8') as mf:
                json.dump(meta, mf)
        except Exception as e:
            print(f"[BlockWriter][Warn] Could not write meta for file {file_index}: {e}")
        if file_index % 50 == 0:
            avg_fill = total_tokens_before_padding / (total_blocks * block_size) if total_blocks else 0.0
            print(f"[BlockWriter] Saved file {file_index} containing {len(buffer_blocks)} blocks | cumulative blocks={total_blocks} avg_fill={avg_fill:.3f}")
        file_index += 1
        buffer_blocks = []
        buffer_block_metas = []

    # Main pass: each packed segment should already be <= block_size.
    for seg_idx, seg_text in enumerate(ds.segments):
        enc = tokenizer.encode(seg_text)
        ids = enc.ids if hasattr(enc, 'ids') else enc
        seg_len = len(ids)
        if seg_len > block_size:
            # Fallback oversize handling (should be rare if packing_max_len==block_size)
            policy = config.get("oversize_policy", "split")
            if policy == "truncate":
                ids = ids[:block_size]
                seg_len = len(ids)
                oversize_truncated_segments += 1
            elif policy == "skip":
                oversize_segments += 1
                continue
            elif policy == "split":  # split into multiple full blocks
                start = 0
                while start < len(ids):
                    piece = ids[start:start+block_size]
                    piece_len = len(piece)
                    if piece_len < block_size:  # pad tail piece
                        piece = piece + [pad_id] * (block_size - piece_len)
                        padded_blocks += 1
                    total_tokens_before_padding += piece_len
                    buffer_blocks.append(piece)
                    buffer_block_metas.append({
                        "seg_idx": seg_idx,
                        "original_seg_len": len(ids),
                        "piece_len": piece_len,
                        "padded": piece_len < block_size,
                        "split_piece": True
                    })
                    total_blocks += 1
                    oversize_split_segments += 1
                    if len(buffer_blocks) >= blocks_per_file:
                        flush_file()
                    start += block_size
                continue
            else:  # raise
                raise ValueError(f"Oversize segment encountered seg_idx={seg_idx} len={seg_len} > block_size={block_size}")

        # Normal case: seg_len <= block_size
        block_tokens = ids.copy()
        if seg_len < block_size:
            block_tokens.extend([pad_id] * (block_size - seg_len))
            padded_blocks += 1
        total_tokens_before_padding += seg_len
        buffer_blocks.append(block_tokens)
        buffer_block_metas.append({
            "seg_idx": seg_idx,
            "seg_len": seg_len,
            "padded": seg_len < block_size,
            "split_piece": False
        })
        total_blocks += 1
        if len(buffer_blocks) >= blocks_per_file:
            flush_file()

    # Final flush
    flush_file(final=True)

    avg_fill = total_tokens_before_padding / (total_blocks * block_size) if total_blocks else 0.0
    print("[BlockWriter][Summary]")
    print(f"  Blocks emitted: {total_blocks}")
    print(f"  Avg fill ratio (pre-pad): {avg_fill:.4f}")
    print(f"  Padded blocks: {padded_blocks}")
    print(f"  Oversize segments (skipped): {oversize_segments}")
    print(f"  Oversize split pieces emitted: {oversize_split_segments}")
    print(f"  Oversize truncated segments: {oversize_truncated_segments}")
    print(f"  Files written: {file_index}")
    return total_blocks

def check_chunk_file(path):
    """Check if a single chunk file is valid. Returns (path, is_valid, error_msg)"""
    try:
        torch.load(path, map_location="cpu")  # still weights_only=True
        return (path, True, None)
    except Exception as e:  # Catch all exceptions instead of specific ones
        return (path, False, str(e))

def sanitize_chunks_fast(config, max_workers=None):
    """Fast parallel sanitization using all available cores."""
    cache_path = config["cache_path"]
    chunk_paths = sorted(glob.glob(os.path.join(cache_path, "chunk*.pt")))
    print(f"Found {len(chunk_paths)} chunk files to check...")
    
    if not chunk_paths:
        print("No chunk files found!")
        return 0, 0
    
    # Use all cores if not specified
    if max_workers is None:
        max_workers = min(32, mp.cpu_count())
    
    print(f"Using {max_workers} parallel workers...")
    
    corrupted_files = []
    valid_files = []
    
    # Use ProcessPoolExecutor for true parallelism
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from tqdm import tqdm
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_path = {executor.submit(check_chunk_file, path): path for path in chunk_paths}
        
        # Process results with progress bar
        for future in tqdm(as_completed(future_to_path), total=len(chunk_paths), desc="Checking files"):
            path, is_valid, error_msg = future.result()
            
            if is_valid:
                valid_files.append(path)
            else:
                print(f"\nCorrupted file: {os.path.basename(path)} - {error_msg}")
                corrupted_files.append(path)
    
    print(f"\nSanitization complete:")
    print(f"  Valid files: {len(valid_files)}")
    print(f"  Corrupted files: {len(corrupted_files)}")
    
    if corrupted_files:
        print(f"\nRemoving {len(corrupted_files)} corrupted files...")
        
        # Sequential deletion is fast enough for small numbers
        for path in corrupted_files:
            try:
                os.remove(path)
                print(f"  Removed: {os.path.basename(path)}")
            except OSError as e:
                print(f"  Failed to remove {os.path.basename(path)}: {e}")
        print("Cleanup complete!")
    else:
        print("No corrupted files found!")
    
    return len(valid_files), len(corrupted_files)

def main():
    parser = argparse.ArgumentParser(description="Tokenize and prepare data for training")    
    parser.add_argument("--config_path", type=str, required=True, help="Path to the configuration file")
    # sanitize flag
    parser.add_argument(
        "--sanitize",
        action="store_true",
        help="Sanitize chunks in the cache directory"
    )
    # sentence-pack flag retained for backward compat but ignored (always on)
    parser.add_argument("--sentence-pack", action="store_true", help="(Deprecated) Enable sentence-aware packing before chunking (now always on)")
    args = parser.parse_args()
    with open(args.config_path, "r") as config_file:
        config = json.load(config_file)
    tokenizer = Tokenizer.from_file(config.get("tokenizer_path"))
    tokenizer.model_max_length = config.get("max_position_embeddings", 512)
  
    if args.sanitize:
        # Sanitize the chunks in the cache directory
        print(f"Sanitizing chunks...")
        valid_count, corrupted_count = sanitize_chunks_fast(config, 100)
        print(f"Valid chunks: {valid_count}, Corrupted chunks removed: {corrupted_count}")
        # tokenizer.save_pretrained(config["cache_path"])  # Save tokenizer to cache path
        return
    
    prepare_data(
        config, tokenizer, config["cache_path"], sentence_pack=args.sentence_pack
    )

if __name__ == "__main__":
     # Force the start method to 'spawn' to avoid deadlocks with transformers tokenizers
    # This is crucial for robust multiprocessing with complex libraries.
    mp.set_start_method("spawn", force=True)
    main()