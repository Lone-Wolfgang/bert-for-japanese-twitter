from transformers import BertTokenizerFast
from tqdm import tqdm
from process_twarc.util import concat_dataset, get_all_files

def train_tokenizer(data_dir,
                    output_dir,
                    checkpoint: str = "cl-tohoku/bert-base-japanese",
                    additional_special_tokens: list=["[URL]", "[USER]"],
                    masks: list=["low_freq_char", "duplicate", "pattern"],
                    tokenizer_class=BertTokenizerFast,
                    vocab_size=100_000):
    
    file_paths = get_all_files(data_dir)
    dataset = concat_dataset(
        file_paths, 
        output_type="Dataset", 
        columns=["text"], 
        masks=masks)
    
    def batch_iterator(batch_size=10000):
        for i in tqdm(range(0, len(dataset), batch_size)):
            yield dataset[i : i + batch_size]["text"]
            
    tokenizer = tokenizer_class.from_pretrained(checkpoint)
    print(f"\n{tokenizer_class}, {checkpoint} loaded.\n")

    if additional_special_tokens:
        tokenizer.add_tokens(additional_special_tokens, special_tokens=True)
        tokenizer.additional_special_tokens = additional_special_tokens
        print(f"\nSpecial Tokens {additional_special_tokens} added to vocabulary.\n")

    bert_tokenizer = tokenizer.train_new_from_iterator(text_iterator=batch_iterator(), vocab_size=vocab_size)
    bert_tokenizer.save_pretrained(output_dir)
    print(f"\nNew Tokenizer saved to {output_dir}.\n")
    return