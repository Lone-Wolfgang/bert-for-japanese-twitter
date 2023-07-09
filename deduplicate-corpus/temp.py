from process_twarc.util import load_parquet
import random

path = "objectives/deduplicate-corpus/intermediate/duplicate-text.parquet"

duplicate_text = load_parquet(path)
print(duplicate_text)