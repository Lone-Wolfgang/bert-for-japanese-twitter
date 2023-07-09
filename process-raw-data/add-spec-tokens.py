from process_twarc.util import save_to_parquet, get_all_files, load_parquet, get_output_path
from modules.process_raw_data import add_special_tokens
from tqdm import tqdm

def process(file_paths, output_dir):
    for file_path in tqdm(file_paths, desc = "Adding special tokens"):
        dataset = load_parquet(file_path)
        output_path = get_output_path(file_path, output_dir)

        for idx, row in dataset.iterrows():
           row["text"] = add_special_tokens(row["text"])
        
        save_to_parquet(dataset, output_path)
    return

paths = {
    "read": "data/tweets/1-base",
    "write": "data/tweets/2-spec-tokens"
}
read, write = paths.values()
file_paths = get_all_files(read)

process(file_paths, write)