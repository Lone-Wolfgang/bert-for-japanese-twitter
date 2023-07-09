from modules.process_raw_data import generate_masks, compile_duplicate_ids
from process_twarc.util import  get_all_files, load_parquet
from tqdm import tqdm


def load_duplicate_text(path_to_duplicates):
    return set(load_parquet(path_to_duplicates)["text"])

paths = {
    "tokenized_datasets": "data/tweets/3-tokenized",
    "duplicate_text": "objectives/deduplicate-corpus/intermediate/duplicate-text.parquet",
    "output_dir": "data/tweets/4-masked",
    "duplicate_ids_output": "objectives/process-raw-data/intermediate/duplicate-ids.parquet"
}
data_dir, path_to_duplicates, output_dir, duplicate_ids_outpt = paths.values()

file_paths = get_all_files(data_dir)

print ("Loading duplicate text. . .")
duplicate_text = load_duplicate_text(path_to_duplicates)
print("Loaded!")

for file_path in tqdm(file_paths):
    masked_dataset = generate_masks(file_path, duplicate_text, output_dir)
    print(f"Len Dataset: {len(masked_dataset)}")
    print(f"Duplicates: {sum(masked_dataset['duplicate'])}")
    print(f"Low Freq Char: {sum(masked_dataset['low_freq_char'])}")
    print(f"Patterns: {sum(masked_dataset['pattern'])}")

compile_duplicate_ids(output_dir, duplicate_ids_outpt)