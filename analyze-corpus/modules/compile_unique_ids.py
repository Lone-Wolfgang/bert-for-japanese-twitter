from process_twarc.util import concat_dataset, get_all_files, save_to_parquet
import pandas as pd


def compile_tweet_ids(base_dir, output_dir):
    base_files = get_all_files(base_dir)

    data = concat_dataset(base_files, "Dataset", columns = "tweet_id")
    compile = lambda column: pd.DataFrame({column: sorted(set(data[column]))})

    ids = compile("tweet_id")
    path_to_output = f"{output_dir}/tweet_ids.parquet"
    save_to_parquet(ids, path_to_output)
    print(f"tweet_ids saved to {path_to_output}")
    return

def compile_user_and_place_ids(rich_dir, output_dir):

    id_columns = ["user_id", "place_id"]
    rich_files = get_all_files(rich_dir)
    
    data = concat_dataset(rich_files, "Dataset", columns = id_columns)
    compile = lambda column: pd.DataFrame({column: sorted(set(data[column]))})

    for column in id_columns:
        print(f"Compiling {column}s.")
        ids = compile(column)
        path_to_output = f"{output_dir}/{column}s.parquet"
        save_to_parquet(ids, path_to_output)
        print(f"{column} saved to {path_to_output}.\n")
    return
        
