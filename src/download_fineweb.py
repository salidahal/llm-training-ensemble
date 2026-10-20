"""
FineWeb-Edu (sample-10BT) -> fixed .bin shards for LM training

Outputs (non-overlapping splits by document):
  - val  : 1 shard of 4,000,000 tokens
  - test : 1 shard of 4,000,000 tokens
  - train: 4 shards of 50,000,000 tokens each

Format: custom .bin
  header: 256 int32
    header[0] = 20240520 (magic)
    header[1] = 1        (version)
    header[2] = ntok     (number of uint16 tokens following)
  payload: ntok uint16 tokens

Run:
  python fineweb.py
"""

import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

# ----------------------------
# Config
# ----------------------------
OUT_DIR = "edu_fineweb_fixed"
REMOTE_NAME = "sample-10BT"

TRAIN_SHARDS = 4
TRAIN_SHARD_SIZE = 50_000_000
VAL_SHARDS = 1
VAL_SHARD_SIZE = 4_000_000
TEST_SHARDS = 1
TEST_SHARD_SIZE = 4_000_000

MAGIC = 20240520
VERSION = 1

DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), OUT_DIR)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# ----------------------------
# Tokenizer
# ----------------------------
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens["<|endoftext|>"]  # 50256

def tokenize(doc):
    # Prefix with EOT as document delimiter
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(doc["text"]))
    arr = np.asarray(tokens, dtype=np.int64)
    if not ((0 <= arr).all() and (arr < 2**16).all()):
        raise ValueError("token id out of range for uint16")
    return arr.astype(np.uint16)

def write_bin(filename, tokens_uint16: np.ndarray):
    tokens_uint16 = np.asarray(tokens_uint16, dtype=np.uint16)
    header = np.zeros(256, dtype=np.int32)
    header[0] = MAGIC
    header[1] = VERSION
    header[2] = tokens_uint16.size
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(tokens_uint16.tobytes())

# ----------------------------
# Split plan
# ----------------------------
# We fill sequentially: val -> test -> train shards
split_plan = [
    ("val",  VAL_SHARDS,  VAL_SHARD_SIZE),
    ("test", TEST_SHARDS, TEST_SHARD_SIZE),
    ("train", TRAIN_SHARDS, TRAIN_SHARD_SIZE),
]

def main():
    # Load dataset (sequential stream order)
    fw = load_dataset("HuggingFaceFW/fineweb-edu", name=REMOTE_NAME, split="train", streaming=True)

    split_i = 0
    shard_i_in_split = 0

    split_name, split_num_shards, shard_target = split_plan[split_i]

    # preallocate shard buffer
    buf = np.empty((shard_target,), dtype=np.uint16)
    fill = 0

    pbar = tqdm(total=shard_target, unit="tokens",
                desc=f"{split_name} shard {shard_i_in_split+1}/{split_num_shards}")

    def finalize_current_shard():
        nonlocal split_name, shard_i_in_split, shard_target, buf, fill, pbar, split_i, split_num_shards
        out_path = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split_name}_{shard_i_in_split:06d}.bin")
        write_bin(out_path, buf)  # EXACT size shard_target
        pbar.close()

        shard_i_in_split += 1
        if shard_i_in_split >= split_num_shards:
            split_i += 1
            if split_i >= len(split_plan):
                return False  # done
            shard_i_in_split = 0
            split_name, split_num_shards, shard_target = split_plan[split_i]

        # reset buffer for next shard
        buf = np.empty((shard_target,), dtype=np.uint16)
        fill = 0
        pbar = tqdm(total=shard_target, unit="tokens",
                    desc=f"{split_name} shard {shard_i_in_split+1}/{split_num_shards}")
        return True

    for doc in fw:
        tokens = tokenize(doc)
        pos = 0
        doc_len = tokens.size

        while pos < doc_len:
            remaining = shard_target - fill
            take = min(remaining, doc_len - pos)

            buf[fill:fill + take] = tokens[pos:pos + take]
            fill += take
            pos += take
            pbar.update(take)

            if fill == shard_target:
                completed_split_name = split_name
                cont = finalize_current_shard()
                if not cont:
                    print("Done. Wrote val/test/train shards.")
                    import sys
                    sys.exit(0)

                if completed_split_name in ("val", "test") and pos < doc_len:
                    pos = doc_len
                    break

    raise RuntimeError("Dataset ended before all requested shards were filled.")

if __name__ == "__main__":
    main()