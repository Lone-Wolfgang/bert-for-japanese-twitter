from modules.compile_unique_ids import compile_tweet_ids, compile_user_and_place_ids

paths = {
    "base_dir": "data/tweets/3-tokenized",
    "rich_dir": "data/tweets/1-rich/",
    "output_dir": "data/corpus-analysis/ids"
}
base_dir, rich_dir, output_dir = paths.values()

compile_tweet_ids(base_dir, output_dir)
compile_user_and_place_ids(rich_dir, output_dir)