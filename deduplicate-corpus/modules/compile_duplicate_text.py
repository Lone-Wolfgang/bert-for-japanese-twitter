import pandas as pd
import os
from tqdm import tqdm
from process_twarc.util import save_to_parquet

def compile_duplicate_text (deduped_dir, path_to_output):
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
    1. From mask.jsonl, get mask_idx, a list of indexes that are flagged as duplicate.

    2. From subcorpus.jsonl, generate a sorted set of duplicate text.

    3. Compile the results of each chunk, resulting in the directory structure below:
    save_directory
    ├── chunk001
    │    |── duplicate-text.txt  <- output
    │    |── subcorpus.jsonl
    │    └── Deduplicated
    │        |── mask.jsonl
    │        └── config.json
    . . .

    4. Compile all duplicate text into a single Parquet file.
    
    """
    chunks = os.listdir(deduped_dir)
    paths = {
        "subcorpus" : lambda chunk: f"{deduped_dir}/{chunk}/subcorpus.jsonl", 
        "mask" : lambda chunk: f"{deduped_dir}/{chunk}/Deduplicated/mask.jsonl",
        "text": lambda chunk: f"{deduped_dir}/{chunk}/duplicate-text.txt"
    }
    path_to_subcorpus, path_to_mask, path_to_text = paths.values()

    def read_jsonl(path):
        return pd.read_json(path, lines=True, encoding="utf-8")
    
    def get_duplicate_idx(mask):
        return mask[mask["duplicate"]].index
    
    def get_duplicate_text(subcorpus, duplicate_idx):
        return subcorpus[subcorpus.index.isin(duplicate_idx)]["text"]
    
    def save_text(duplicate_text, path_to_text):
        """Saves a batch of duplicate text to a text file."""
        duplicate_text = sorted(set(duplicate_text))
        duplicate_text = "\n".join(duplicate_text)
        with open(path_to_text, "w", encoding="utf-8") as f:
            f.write(duplicate_text)
    
    def load_text(path_to_text):
        with open(path_to_text, "r", encoding="utf-8") as f:
            return set(f.read().split("\n"))

    for chunk in tqdm(chunks, desc = "Unpacking chunks."):
        if os.path.exists(path_to_text(chunk)):
            pass
        else:
            subcorpus = read_jsonl(path_to_subcorpus(chunk))
            mask = read_jsonl(path_to_mask(chunk))

            duplicate_idx = get_duplicate_idx(mask)
            duplicate_text = get_duplicate_text(subcorpus, duplicate_idx)
            save_text(duplicate_text, path_to_text(chunk))

    existing_duplicate_text = set()                         
    for chunk in tqdm(chunks, desc="Compiling duplicate text"):
        duplicate_text = load_text(path_to_text(chunk))
        existing_duplicate_text = existing_duplicate_text.union(duplicate_text)
    
    save_text(existing_duplicate_text, path_to_output)
    return


    


