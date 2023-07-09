from modules.train_tokenizer import train_tokenizer

paths = {
    "data_dir": "data/tweets/4-masked",
    "output_dir": "data/tokenizers/vocab-spring"
}
data_dir, output_dir = paths.values()

train_tokenizer(data_dir, output_dir)