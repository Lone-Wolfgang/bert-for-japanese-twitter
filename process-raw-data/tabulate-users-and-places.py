from modules.process_raw_data import flatten_users_and_places, tabulate_user_data, tabulate_place_data, tabulate_process, count_stats
from process_twarc.util import get_remaining_files
from tqdm import tqdm

def flatten_process(file_paths, intermediate_dir):
    for file_path in tqdm(file_paths, desc="Unpacking user and place data"):
        flatten_users_and_places(file_path, intermediate_dir)

paths = {
    "data_dir": "data/tweets/0-raw",
    "intermediate_dir": "objectives/process-raw-data/intermediate",
    "rich_dir": "data/tweets/1-rich",
    "output_dir": "data/corpus-analysis"
}
data_dir, intermediate_dir, rich_dir, output_dir = paths.values()

file_paths = get_remaining_files(data_dir, intermediate_dir)

flatten_process(file_paths, intermediate_dir)
count_stats(rich_dir, intermediate_dir)
tabulate_process(tabulate_user_data, intermediate_dir, intermediate_dir, output_dir )
tabulate_process(tabulate_place_data, intermediate_dir, intermediate_dir, output_dir )