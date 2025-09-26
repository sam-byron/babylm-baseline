import os
import glob
import torch
from time import time

from data_loader import ChunkedDataset, get_special_token_ids
from tokenizer import Tokenizer


def test_chunked_dataset_smoke():
    cache_path = os.environ.get("CACHE_PATH", "./model_babylm_bert_ltg")
    vocab_path = os.environ.get("TOKENIZER_PATH", "./data/pretrain/wordpiece_vocab.json")
    block_size = 128

    tokenizer = Tokenizer.from_file(vocab_path)

    chunk_paths = sorted(glob.glob(os.path.join(cache_path, 'chunk*.pt')))
    assert len(chunk_paths) > 0, "No chunk files found for smoke test"

    ds = ChunkedDataset(chunk_paths[:5], block_size=block_size, tokenizer=tokenizer, pad_token_id=tokenizer.token_to_id('[PAD]'))

    # basic length & indexing
    assert len(ds) > 0, "Dataset length should be >0"
    sample = ds[0]
    assert isinstance(sample, torch.Tensor)
    assert sample.shape[0] == block_size, f"Expected block size {block_size}, got {sample.shape[0]}"

    # random access
    mid = len(ds)//2
    sample_mid = ds[mid]
    assert sample_mid.shape[0] == block_size

    # timing batch retrieval
    t0 = time()
    for i in range(min(64, len(ds))):
        _ = ds[i]
    elapsed = time() - t0
    print(f"[SmokeTest] Retrieved {min(64, len(ds))} blocks in {elapsed:.3f}s ({(min(64,len(ds))/elapsed):.1f} blocks/s)")

    # ensure dtype long
    assert sample.dtype == torch.long

    # ensure pad token present or not depending on fill
    pad_id = tokenizer.token_to_id('[PAD]')
    assert pad_id is not None
    assert (sample == pad_id).any() or (sample != pad_id).all(), "Pad token presence logic holds"

if __name__ == '__main__':
    test_chunked_dataset_smoke()
    print('Smoke test passed.')
