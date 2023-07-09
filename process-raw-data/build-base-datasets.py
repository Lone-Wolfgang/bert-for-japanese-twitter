from process_twarc.util import get_all_files
from modules.process_raw_data import process_twarc_file
from tqdm import tqdm

def build_base_datasets(file_paths, output_dir):
    for file_path in tqdm(file_paths, desc = "Building base dataset"):
        process_twarc_file(file_path, output_dir)
    return

paths = {
    "raw_data": "data/tweets/0-raw",
    "output": "data/tweets/1-base"
}
data_dir, output_dir = paths.values()
file_paths = get_all_files(data_dir)

build_base_datasets(file_paths, output_dir)

