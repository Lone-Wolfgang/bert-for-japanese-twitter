from datetime import datetime
import pytz
import pandas as pd
from ntpath import basename
from process_twarc.util import get_file_type, get_output_path, save_to_parquet, load_dataset, save_dict, load_dict, get_all_files, concat_dataset
import re
from typing import Tuple, Dict, Any, List
from tqdm import tqdm

def process_twarc_file(file_path: str, output_dir: str):
    """
    Process a file acquired from Twarc2 and generate a parquet file with columns: tweet_id, text.

    Args:
        file_path (str): Path to the input file.
        output_dir (str): Directory where the generated parquet file will be saved.
    """
    file_type = get_file_type(file_path)

    if file_type == "jsonl":
        df = pd.read_json(file_path, lines=True, encoding="utf-8")
    elif file_type == "json" or file_type == "txt":
        df = pd.read_json(file_path, encoding="utf-8")
    else:
        raise ValueError("Unsupported file type: {}".format(file_type))

    if file_type == "txt":
        tweet_ids = df["id"].astype(str).tolist()
        texts = df["text"].tolist()
    else:
        tweet_ids = []
        texts = []

        for data in df["data"]:
            for item in data:
                tweet_ids.append(str(item["id"]))
                texts.append(item["text"])

    new_df = pd.DataFrame({"tweet_id": tweet_ids, "text": texts})

    output_path = get_output_path(file_path, output_dir, file_type="parquet")
    save_to_parquet(new_df, output_path)

    return new_df



def add_special_tokens(text: str):
    """
    Replaces URLs and usernames in the given text with special tokens.

    Args:
        text (str): The input text.

    Returns:
        str: The text with replaced URLs and usernames.
    """
    text = re.sub("https://t[^ ]+", "[URL]", text)
    text = re.sub("@[^ ]+", "[USER]", text)
    return text

def convert_utc_to_jct(utc_datetime_string):
    utc_datetime = datetime.strptime(utc_datetime_string, "%Y-%m-%dT%H:%M:%S.%fZ")
    
    # Set the timezone to UTC
    utc_timezone = pytz.timezone('UTC')
    utc_datetime = utc_timezone.localize(utc_datetime)
    
    # Convert the datetime to JCT (Japan Standard Time)
    jct_timezone = pytz.timezone('Asia/Tokyo')
    jct_datetime = utc_datetime.astimezone(jct_timezone)
    
    # Format the JCT datetime as a string without milliseconds
    jct_datetime_string = jct_datetime.strftime("%Y-%m-%dT%H:%M:%S")
    
    return jct_datetime_string

import pandas as pd


def build_tokenized_dataset(file_path:str,
                            tokenizer:object,
                            output_dir:str,):
    """
    The tokenized dataset will be used to speed-up the deduplication process.

    This process will tokenize the dataset, decode it, and return a token-split string.

    Tokenized strings are passed to the deduplicator, which proceeeds to tokenize strings
    by the rapid word-split method.

    Args
        file_path (str): Path to the file to be tokenized.
        output_dir (str): Path to the directory where the file is to be saved.

    Returns
        Parquet table with the columns:
            "tweet_id" (str): unique value assigned to the tweet by Twitter,
            "text" (str): text with URLs and usernames masked
            "tokenized" (str): tokenized, decoded, return as token-split string.
                
                *[CLS] and [SEP] tokens retained. "##" wordpiece delimiting removed.

                Ex:
                Text:
                [USER] 明日は祝日なんですね～😅気づかなかった～😅今日もがんばりましょ～💪

                Tokenized:
                [CLS] [USER] 明日 は 祝日 な ん です ね ～ 😅 気づか なかっ た ～ 😅 今日 も がんばり ましょ ～ 💪 [SEP]
                

    """
    dataset = load_dataset(file_path)
    tokenize_function = lambda dataset: tokenizer(dataset["text"])
    tokenized_dataset = dataset.map(tokenize_function)
    decode_function = lambda dataset: {"tokenized": tokenizer.decode(dataset["input_ids"])}
    tokenized_dataset = tokenized_dataset.map(decode_function)
    
    output_path = get_output_path(file_path, output_dir)
    save_to_parquet(tokenized_dataset, output_path)
    return

def generate_masks(file_path, duplicate_text, output_dir):
    """
    Generates masks based on various conditions for a given dataset.

    Args:
        file_path (str): Path to the dataset file.
        duplicate_text (Set[str]): Set of duplicate texts to check against.
        output_dir (str): Directory to save the output.

    Returns:
        pd.DataFrame: Processed dataset with added mask columns.

    """
    def check_duplicate(example:str, duplicate_text:set):
        """Checks if the text has been flagged as duplicate."""
        return example in duplicate_text

    def check_low_freq_char(example:str):
        """For training the tokenizer, checks if the dataset has a low frequency character."""
        return "[UNK]" in example
 
    def check_pattern(example:str):
        """Check if the text has one of the frequently occuring patterns defined below."""
        patterns = [
            "\AI('m| was) at.*in.*",
            "\(@ .*in.*\)"
            ]
        return any(re.search(pattern, example) for pattern in patterns)

    dataset = load_dataset(file_path)
    dataset["low_freq_char"], dataset["duplicate"], dataset["pattern"] = False, False, False
    for idx, row in dataset.iterrows():
        text = row["text"]
        tokenized = row["tokenized"]
        dataset.at[idx, "low_freq_char"] = check_low_freq_char(tokenized)
        dataset.at[idx, "duplicate"] = check_duplicate(tokenized, duplicate_text)
        dataset.at[idx, "pattern"] = check_pattern(text)  
    
    output_path = get_output_path(file_path, output_dir)
    save_to_parquet(dataset, output_path)
    return dataset

def compile_duplicate_ids(masked_dir, path_to_output):
    file_paths = get_all_files(masked_dir)
    dataset = concat_dataset(file_paths, columns = ["tweet_id", "duplicate"])
    duplicate_ids = dataset[dataset["duplicate"]]
    duplicate_ids = duplicate_ids.drop(columns="duplicate").reset_index(drop=True)
    duplicate_ids.rename(columns={"tweet_id":"duplicate_id"}, inplace=True)
    save_to_parquet(duplicate_ids, path_to_output)
    print(f"Duplicate IDs compiled. Saved to {path_to_output}")
    return

def build_rich_dataset(file_path: str, duplicate_ids: set, output_dir: str):
    """
    Given a path to a jsonl file packaged by Twarc, prepare a "rich" dataset with
    the following columns:
    
    Args:
        file_path: Path to the Twarc jsonl file.
        output_dir (str): Path to the directory where the file is to be saved.
    
    Returns:
        Paruqet table, with the columns:
            "tweet_id" (str): unique value assigned to the tweet by Twitter,
            "text" (str): text of the tweet, users and urls masked,
            "conversation_id" (str): tweet_id of the tweet at the root of the coversation tree,
            "place_id" (str): unique value assigned to place where the Tweet was made,
            "has_url" (bool): whether or not the Tweet has a url,
            "has_mention" (bool): whether or not the Tweet mentions another user,
            "has_hashtag" (bool): whether or not the Tweet has a hashtag
            "duplicate" (bool): whether or not the Tweet has been flagged as a duplicate
    """
    file_type = get_file_type(file_path)
    old_df = None
    if file_type == "json":
        old_df = pd.read_json(file_path, encoding="utf-8")
    elif file_type == "jsonl":
        old_df = pd.read_json(file_path, lines=True, encoding="utf-8")

    if old_df is not None:
        new_df = {
            "tweet_id": [],
            "text": [],
            "conversation_id": [],
            "user_id": [],
            "created_at": [],
            "place_id": [],
            "possibly_sensitive": [],
            "has_url": [],
            "has_mention": [],
            "has_hashtag": [],
            "duplicate": []
        }

        for i in range(len(old_df["data"])):
            for j in range(len(old_df["data"][i])):
                data = old_df["data"][i][j]

                tweet_id = str(data["id"])
                text = add_special_tokens(data["text"])
                conversation_id = data["conversation_id"]
                user_id = data["author_id"]
                created_at = convert_utc_to_jct(data["created_at"])
                if "geo" in data.keys() and "place_id" in data["geo"].keys():
                    place_id = data["geo"]["place_id"]
                else:
                    place_id = "None"
                possibly_sensitive = data["possibly_sensitive"]

                has_url, has_mention, has_hashtag = False, False, False
                if "entities" in data.keys():
                    entities = data["entities"]
                    if "urls" in entities.keys():
                        has_url = True
                    if "mentions" in entities.keys():
                        has_mention = True
                    if "hashtags" in entities.keys():
                        has_hashtag = True

                duplicate = tweet_id in duplicate_ids

                new_df["tweet_id"].append(tweet_id)
                new_df["text"].append(text)
                new_df["conversation_id"].append(conversation_id)
                new_df["user_id"].append(user_id)
                new_df["created_at"].append(created_at)
                new_df["place_id"].append(place_id)
                new_df["possibly_sensitive"].append(possibly_sensitive)
                new_df["has_url"].append(has_url)
                new_df["has_mention"].append(has_mention)
                new_df["has_hashtag"].append(has_hashtag)
                new_df["duplicate"].append(duplicate)

        new_df = pd.DataFrame(new_df)
        output_path = get_output_path(file_path, output_dir, file_type="parquet")
        save_to_parquet(new_df, output_path)
        return

def flatten_users_and_places(file_path: str, output_dir: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Extracts the "includes" column from a JSON or JSONL file, condenses batches into a simple dictionary,
    and returns separate dictionaries for users and places.

    Args:
        file_path (str): The path to the JSON or JSONL file.
        output_dir (str): The path to the directory where dictionaries are to be saved.


    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]: A tuple containing two dictionaries:
            - user_dict: A dictionary mapping user IDs to user data.
            - place_dict: A dictionary mapping place IDs to place data.
    """

    file_type = get_file_type(file_path)
    if file_type == "jsonl":
        df = pd.read_json(file_path, lines=True, encoding="utf-8")
    elif file_type == "json":
        df = pd.read_json(file_path, encoding="utf-8")
    else:
        df = None

    if isinstance(df, pd.DataFrame):
        includes = df["includes"]
        user_dict = {}
        place_dict = {}

        for i in range(len(includes)):
            users = includes[i]["users"]
            places = includes[i]["places"]

            user_id2data = {k: v for k, v in zip([user.pop("id") for user in users], users)}
            place_id2data = {k: v for k, v in zip([place.pop("id") for place in places], places)}
            user_dict.update(user_id2data)
            place_dict.update(place_id2data)
        
        user_output_path = get_output_path(file_path, f"{output_dir}/users")
        place_output_path = get_output_path(file_path, f"{output_dir}/places")

        save_dict(user_dict, user_output_path)
        save_dict(place_dict, place_output_path)
        return
    
def count_stats(rich_dir, output_dir):
    id_columns = ["user_id", "place_id"]

    target_columns = lambda id_column: [id_column, "possibly_sensitive", "has_url", "has_mention", "has_hashtag", "duplicate"]
    rich_files = get_all_files(rich_dir)

    for id_column in id_columns:
        if id_column == "user_id":
            path_to_output = f"{output_dir}/user-stats.json"
        elif id_column == "place_id":
            path_to_output = f"{output_dir}/place-stats.json"
        
        stats = {}
        rich_data = concat_dataset(rich_files, columns=target_columns(id_column))
        grouped_data = rich_data.groupby(id_column)
        for id_, group in tqdm(grouped_data, desc="Computing sample_tweet_count, possibly_sensitive_count, url_count, mention_count, hashtag_count, duplicate_count"):
            stats.update(
                {id_:{
                    "sample_tweet_count": len(group),
                    "possibly_sensitive_count": sum(group["possibly_sensitive"]),
                    "url_count": sum(group["has_url"]),
                    "mention_count": sum(group["has_mention"]),
                    "hashtag_count": sum(group["has_hashtag"]),
                    "duplicate_count": sum(group["duplicate"])
            }})
        
        save_dict(stats, path_to_output)
        print (f"Stats saved to {path_to_output}.")
    return

def filter_dictionary(dictionary:dict, target_keys:(str|List[str])):
    """"
    The inlcudes column contains more users and places then are present in the dataset.

    Given a dictionary, and a list of target keys, reduce the dictionary down to entries
    with the target keys.

    Args:
        dictionary(dict): Dictionary to be filtered.
        target_keys( (str|list(str)) ): List of keys to be included in the filtered dictionary.

    Returns:
        filtered_dictionary(dict): Dictionary with entries for only the target keys. 
    """
    return {key: dictionary[key] for key in target_keys if key in dictionary}

def tabulate_user_data(user_dict, user_ids, user_stats):
    user_dict = filter_dictionary(user_dict, user_ids)

    user_table = {
        "user_id": [],
        "username": [],
        "link": [],
        "created_at": [],
        "followers_count": [],
        "following_count": [],
        "listed_count": [],
        "lifetime_tweet_count": [],
        "lifetime_day_count": [],
        "lifetime_tweet_rate": [],
        "sample_tweet_count": [],
        "sample_day_count": [],  
        "sample_tweet_rate": [],
        "possibly_sensitive_count": [],
        "url_count": [],
        "mention_count": [],
        "hashtag_count": [],
        "duplicate_count": []
    }

    sample_end_date = datetime(2023, 5, 8)  # May 8, 2023
    sample_start_date = datetime(2022, 6, 12)  # June 12, 2022

    for user_id in user_dict.keys():
        data = user_dict[user_id]
        username = data["username"]
        link = f"https://twitter.com/{username}"

        created_at = data["created_at"]
        counts = user_stats[user_id]
        public_metrics = data["public_metrics"]

        followers_count = public_metrics["followers_count"]
        following_count = public_metrics["following_count"]
        lifetime_tweet_count = public_metrics["tweet_count"]
        sample_tweet_count = counts["sample_tweet_count"]
        listed_count = public_metrics["listed_count"]

        possibly_sensitive_count = counts["possibly_sensitive_count"]
        url_count = counts["url_count"]
        mention_count = counts["mention_count"]
        hashtag_count = counts["hashtag_count"]
        duplicate_count = counts["duplicate_count"]

        created_at_date = datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%S.%fZ")
        lifetime_day_count = (sample_end_date - created_at_date).days

        if created_at_date >= sample_start_date:
            sample_day_count = (sample_end_date - created_at_date).days
        else:
            sample_day_count = (sample_end_date - sample_start_date).days

        lifetime_tweet_rate = int(lifetime_tweet_count / lifetime_day_count) if lifetime_day_count > 0 else 0
        sample_tweet_rate = int(sample_tweet_count / sample_day_count) if sample_day_count > 0 else 0

        user_table["user_id"].append(user_id)
        user_table["username"].append(username)
        user_table["link"].append(link)
        user_table["created_at"].append(created_at)
        user_table["followers_count"].append(followers_count)
        user_table["following_count"].append(following_count)
        user_table["listed_count"].append(listed_count)
        user_table["lifetime_tweet_count"].append(lifetime_tweet_count)
        user_table["lifetime_day_count"].append(lifetime_day_count)
        user_table["lifetime_tweet_rate"].append(lifetime_tweet_rate)
        user_table["sample_day_count"].append(sample_day_count)
        user_table["sample_tweet_count"].append(sample_tweet_count)
        user_table["sample_tweet_rate"].append(sample_tweet_rate)
        user_table["possibly_sensitive_count"].append(possibly_sensitive_count)
        user_table["url_count"].append(url_count)
        user_table["mention_count"].append(mention_count)
        user_table["hashtag_count"].append(hashtag_count)
        user_table["duplicate_count"].append(duplicate_count)
        
    user_table = pd.DataFrame(user_table)

    return user_table

def tabulate_place_data(place_dict, place_ids, place_stats):
    place_dict = filter_dictionary(place_dict, place_ids)

    place_table = {
        "place_id": [],
        "full_name": [],
        "name": [],
        "place_type": [],
        "sample_tweet_count": [],
        "possibly_sensitive_count": [],
        "url_count": [],
        "mention_count": [],
        "hashtag_count": [],
        "duplicate_count": [],
        "bbox": []
    }

    for place_id in place_dict.keys():
        counts = place_stats[place_id]

        data = place_dict[place_id]
        full_name = data["full_name"]
        name = data["name"]
        place_type = data["place_type"]

        sample_tweet_count = counts["sample_tweet_count"]
        possibly_sensitive_count = counts["possibly_sensitive_count"]
        url_count = counts["url_count"]
        mention_count = counts["mention_count"]
        hashtag_count = counts["hashtag_count"]
        duplicate_count = counts["duplicate_count"]

        bbox = str(data["geo"]["bbox"])

        place_table["place_id"].append(place_id)
        place_table["full_name"].append(full_name)
        place_table["name"].append(name)
        place_table["place_type"].append(place_type)
        place_table["sample_tweet_count"].append(sample_tweet_count)
        place_table["possibly_sensitive_count"].append(possibly_sensitive_count)
        place_table["url_count"].append(url_count)
        place_table["mention_count"].append(mention_count)
        place_table["hashtag_count"].append(hashtag_count)
        place_table["duplicate_count"].append(duplicate_count )
        place_table["bbox"].append(bbox)

    place_table = pd.DataFrame(place_table)
    return place_table


def tabulate_process(tabulate_method, data_dir: str, stats_dir: str, output_dir: str):

    if tabulate_method == tabulate_user_data:
        path_to_data = f"{data_dir}/users"
        stats = load_dict(f"{stats_dir}/user-stats.json")
        path_to_output = f"{output_dir}/users.csv"

    elif tabulate_method == tabulate_place_data:
        path_to_data = f"{data_dir}/places"
        stats = load_dict(f"{stats_dir}/place-stats.json")
        path_to_output = f"{output_dir}/places.csv"

    print("Compiling id list. . .")
    id_list = set(stats.keys())
    print("ID list compiled!")
    sort_by_basename = lambda file_paths: sorted(file_paths, key=lambda file_path: basename(file_path), reverse=True)
    file_paths = sort_by_basename(get_all_files(path_to_data))
    last_file = file_paths[-1]

    table = pd.DataFrame()
    for file_path in tqdm(file_paths, desc="Tabulating"):
        if id_list:
            data = load_dict(file_path)
            all_ids = list(data.keys())
            target_ids = [id_ for id_ in all_ids if id_ in id_list]
            id_list.difference_update(target_ids)
            chunk = tabulate_method(data, target_ids, stats)

            table = pd.concat([table, chunk])

            if (not id_list) or (file_path==last_file):
                table.to_csv(path_to_output, encoding="utf-8-sig", index=False)

            print(f"Searching for {len(id_list)} remaining IDs.")
        else:
            print("IDs Tabulated.")
            print(f"Total Places: {len(table)}")
            table.to_csv(path_to_output, encoding="utf-8-sig", index=False)
    return