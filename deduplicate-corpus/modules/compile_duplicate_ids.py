import pandas as pd
import os
from tqdm import tqdm
from process_twarc.util import save_to_parquet

def compile_duplicate_ids(dedup_directory: str, path_to_output: str):
    """
    Deduplication results in the following directory tree:

            save_directory
        ├── chunk001
        │    |── subcorpus.jsonl
        │    └── Deduplicated
        │        |── mask.jsonl
        │        └── config.json
        ├── chunk002
        │    |── subcorpus.jsonl
        │    └── Deduplicated
        │        |── mask.jsonl
        │        └── config.json
        ├── chunk003
        │    |── subcorpus.jsonl
        │    └── Deduplicated
        │        |── mask.jsonl
        │        └── config.json
        . . .
        └── chunk{n}
             |── subcorpus.jsonl
             └── Deduplicated
                 |── mask.jsonl
                 └── config.json

    In this procedure, we will:
        1. From the subcorpus.jsonl, generate "dedup_id2tweet_id", which maps tweets as they are identified
        in the deduplicator their tweet ID.

        2. From the mask.json, extract which candidates were identified as duplicates.

        3. Compile the results of each chunk, resulting in the directory structure below:
            save_directory
        ├── chunk001
             |── id-chunk.txt
        │    |── subcorpus.jsonl
        │    └── Deduplicated
        │        |── mask.jsonl
        │        └── config.json

        4. Compile all id-chunks into a single PARQUET file.
        . . .
    """
    def read_jsonl(file_path: str):
        """Reads a JSONL file and returns a DataFrame."""
        return pd.read_json(file_path, lines=True, encoding="utf-8")

    def save_id_chunk(id_chunk, save_path):
        """Saves a chunk of IDs to a text file."""
        id_chunk = "\n".join(id_chunk)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(id_chunk)

    def load_id_chunk(path_to_text):
        """Loads a chunk of IDs from a text file and returns them as a set."""
        with open(path_to_text, "r", encoding="utf-8") as f:
            return set(f.read().split("\n"))

    def get_dedup_id2tweet_id(subcorpus):
        """Creates a dictionary mapping deduplicated IDs to tweet IDs from a DataFrame."""
        dedup_id2tweet_id = {
            k: v for k, v in zip(
                subcorpus['id'].tolist(),
                subcorpus['tweet_id'].tolist())
        }
        return dedup_id2tweet_id

    def get_id_chunk(subcorpus, mask):
        """Extracts a chunk of IDs based on a mask and returns them as a list."""
        dedup_ids = mask[mask['duplicate'] == True]['id'].tolist()
        dedup_id2tweet_id = get_dedup_id2tweet_id(subcorpus)
        id_chunk = [str(dedup_id2tweet_id[id_]) for id_ in dedup_ids]
        return id_chunk

    def process_chunk(chunk_path: str):
        """Processes a chunk by reading necessary files, extracting ID chunks, and saving them."""
        subcorpus_path = os.path.join(chunk_path, "subcorpus.jsonl")
        mask_path = os.path.join(chunk_path, "Deduplicated/mask.jsonl")
        save_path = os.path.join(chunk_path, "id-chunk.txt")

        subcorpus = read_jsonl(subcorpus_path)
        mask = read_jsonl(mask_path)

        id_chunk = get_id_chunk(subcorpus, mask)
        save_id_chunk(id_chunk, save_path)
        return id_chunk

    chunks = os.listdir(dedup_directory)
    for chunk in tqdm(chunks, desc=("Unpacking results")):
        chunk_path = os.path.join(dedup_directory, chunk)
        if os.path.exists(f"{chunk_path}/id-chunk.txt"):
            pass
        else:
            process_chunk(chunk_path)

    duplicate_ids = set()
    for chunk in tqdm(chunks, desc="Compiling duplicate Tweet IDs."):
        id_chunk = load_id_chunk(f"{dedup_directory}/{chunk}/id-chunk.txt")
        duplicate_ids = duplicate_ids.union(id_chunk)

    compiled_df = pd.DataFrame(list(duplicate_ids), columns=["duplicate_ids"])
    save_to_parquet(compiled_df, path_to_output)
    print(f"{len(duplicate_ids)} duplicate ids compiled. Saved to {path_to_output}.")
    return
