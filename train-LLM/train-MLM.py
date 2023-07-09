from modules.train_MLM import train_masked_language_model


paths = {
    "data_dir": "data/tweets/4-masked",
    "output_dir":"data/models/bert-for-japanese-twitter"
}
data_dir, output_dir = paths.values()

train_masked_language_model(data_dir, output_dir)