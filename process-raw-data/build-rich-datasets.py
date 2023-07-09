from modules.process_raw_data import build_rich_dataset
from process_twarc.util import load_parquet, get_remaining_files
from tqdm import tqdm

def load_duplicate_ids(path_to_duplicates):
    return set(load_parquet(path_to_duplicates).iloc[:, 0])

def build_rich_datasets(file_paths, duplicate_ids, output_dir):
    for file_path in tqdm(file_paths, desc = "Building rich datasets"):
        build_rich_dataset(file_path, duplicate_ids, output_dir)
    return

paths = {
    "raw_data": "data/tweets/0-raw",
    "duplicate_ids": "objectives/process-raw-data/intermediate/duplicate-ids.parquet",
    "output": "data/tweets/1-rich"
}
data_dir, path_to_duplicates, output_dir = paths.values()

file_paths= get_remaining_files(data_dir, output_dir)

print("Loading duplicate ids. . .")
duplicate_ids = load_duplicate_ids(path_to_duplicates)


dataset = build_rich_datasets(file_paths, duplicate_ids, output_dir)
