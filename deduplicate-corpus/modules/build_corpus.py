import pandas as pd
import os
from tqdm import tqdm
from process_twarc.util import get_all_files, concat_dataset
import jsonlines

def build_chunked_corpus(data_directory: str, 
                         save_directory: str,
                         sample_frac: float = 1.0, 
                         num_epochs: int = 1, 
                         num_chunks: int = 10,
                         ):
    """
    Deduplication occupies quite a bit of RAM. To mangage memory, this procedure will break the corpus into chunks.

    My experience 32 Gb of RAM was sufficient for 4-5 million tweets, and 64 Gb for 6-7 million Tweets.

    Args:
        data_directory (str): Path to the directory where data is held.
        save_directory (str): Path to the directory where the chunked corpus is to be saved.
        num_epochs (int, optional): Number of times the complete dataset is chunked. Defaults to 1.
        num_chunks (int, optional): Number of chunks into which the corpus is to be divided. Defaults to 10.

    Returns:
        Chunked corpus, formatted as shown below:

        Total chunks equalts the num_epochs * num_chunks

        save_directory
        ├── chunk001
        │   └── subcorpus.jsonl
        ├── chunk002
        │   └── subcorpus.jsonl
        ├── chunk003
        │   └── subcorpus.jsonl
        . . .
        └── chunk{n}
            └── subcorpus.jsonl

    """

    def init_chunk(save_directory):
        """
        Initialize a new chunk directory for saving subcorpus.

        Args:
            save_directory (str): Path to the directory where the chunked corpus is to be saved.

        Returns:
            chunk (str): Name of the newly created chunk directory.
        """
        def get_next_chunk(last_chunk):
            """
            Get the next chunk name based on the last chunk name.

            Args:
                last_chunk (str): Name of the last chunk directory.

            Returns:
                next_chunk (str): Name of the next chunk directory.
            """
            numeric_part = int(last_chunk[5:])
            numeric_part += 1
            next_chunk = "chunk" + str(numeric_part).zfill(3)
            return next_chunk
        
        chunks = os.listdir(save_directory)
        if chunks == []:
            chunk = "chunk001"
        else:
            last_chunk = sorted(chunks, reverse=True)[0]
            chunk = get_next_chunk(last_chunk)
        os.mkdir(f"{save_directory}/{chunk}")
        print(f"Generated folder for {chunk}")
        return chunk

    def get_chunk_idx(dataset, num_chunks):
        """
        Get the indices at which to divide the dataset into chunks.

        Args:
            dataset: The dataset to be divided.
            num_chunks (int): Number of chunks into which the corpus is to be divided.

        Returns:
            chunk_idx (list): List of indices representing the chunk boundaries.
        """
        chunk_size = int(len(dataset) / num_chunks) + 1
        chunk_idx = list(range(0, len(dataset), chunk_size))
        chunk_idx = chunk_idx + [len(dataset)]
        return chunk_idx
    
    def get_path_to_output(chunk):
        """ Path to save the subcorpus within the chunk."""
        return  f"{save_directory}/{chunk}/subcorpus.jsonl"
    
    def chunk_to_jsonl(data_chunk, path_to_output):
        """
        Convert a data chunk to the JSONL format and save it.

        Args:
            data_chunk: The data chunk to be converted and saved.
            save_path (str): Path to save the subcorpus.

        """
        data_chunk = pd.DataFrame(data_chunk).reset_index(drop=True)
        subcorpus = []
        #ERROR CULPRIT 1
        for idx, row in tqdm(data_chunk.iterrows(), desc="Building subcorpus", total=len(data_chunk)):
            example = {
                "id": idx,
                "tweet_id": row["tweet_id"],
                "text": row["tokenized"]
            }
            subcorpus.append(example)

        with jsonlines.open(path_to_output, mode="w") as writer:
            writer.write_all(subcorpus) 

        

    file_paths = get_all_files(data_directory)
    dataset = concat_dataset(file_paths, output_type="Dataset", columns=["tweet_id", "tokenized"])

    for epoch in range(num_epochs):
        print(f"Initiating epoch {epoch + 1}\n")
        sample_size = int(len(dataset) * sample_frac)
        if sample_frac < 1.0:
            print(f"Sampling {sample_size} tweets.")
        #ERROR CULPRIT 2
        dataset = dataset.shuffle().select(range(sample_size))
        print("Corpus compiled and shuffled.")

        chunk_idx = get_chunk_idx(dataset, num_chunks)
        for i in range(num_chunks):
            chunk = init_chunk(save_directory)
            start = chunk_idx[i]
            stop = chunk_idx[i + 1]
            data_chunk = dataset[start:stop]
            path_to_output = get_path_to_output(chunk)
            subcorpus = chunk_to_jsonl(data_chunk, path_to_output)
        return subcorpus