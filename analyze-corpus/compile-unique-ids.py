from modules.compile_unique_ids import compile_unique_ids

paths = {
    "rich_dir": "data/tweets/1-rich/",
    "output_dir": "data/corpus-analysis/ids"
}
rich_dir, output_dir = paths.values()

compile_unique_ids(rich_dir, output_dir)