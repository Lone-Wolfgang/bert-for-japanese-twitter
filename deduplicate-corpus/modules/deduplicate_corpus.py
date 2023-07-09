from nlp_dedup.deduper import Deduper
import os
import jsonlines
import gc

def deduplicate_corpus(
        dir_to_dedup: str,
        ngram_size: int,
        similarity_threshold: float,
        store_corpus_to_disk: bool=False,
        store_lsh_cache_to_disk: bool=False,
):

    def get_chunks(dir_to_dedup):
        return os.listdir(dir_to_dedup)
    
    def load_corpus_from_jsonl(file_path):
        with jsonlines.open(file_path) as reader:
            corpus = [data for data in reader]
        return corpus
    
    chunks = get_chunks(dir_to_dedup)
    for chunk in chunks:
        print(f"Initiating deduplication of {chunk}.")
        paths = {
            "corpus": f"{dir_to_dedup}/{chunk}/subcorpus.jsonl",
            "output_dir": f"{dir_to_dedup}/{chunk}/Deduplicated"
        }
        path_to_corpus, output_dir = paths.values()

        corpus = load_corpus_from_jsonl(path_to_corpus)

        deduper = Deduper(ngram_size=ngram_size,
                          similarity_threshold=similarity_threshold,
                          store_config_to_disk=store_corpus_to_disk,
                          store_lsh_cache_to_disk=store_lsh_cache_to_disk)
        
        print("Deduper initiated.")
        print("ngram_size:", ngram_size)
        print("similarity_threshold", similarity_threshold)
        
        deduper.deduplicate(corpus=corpus,
                            output_dir=output_dir)
        
        del deduper
        del corpus
        gc.collect()
        print(f"Deduplication of {chunk} complete.")
