import os
import pandas as pd
from transformers import Trainer,BertJapaneseTokenizer, AutoModelForMaskedLM,TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
import pyarrow.parquet as pq
from torch.utils.data import DataLoader
from process_twarc.util import get_all_files, concat_dataset 


def train_masked_language_model(
    data_dir:str,
    output_dir:str,
    tokenizer_class: object=BertJapaneseTokenizer,
    path_to_tokenizer: str="LoneWolfgang/bert-base-plus-char",
    model_class: object=AutoModelForMaskedLM,
    path_to_model: str= "cl-tohoku/bert-base-japanese-whole-word-masking",
    collator_class: object=DataCollatorForLanguageModeling):

    def load_raw_datasets(data_dir):
        file_paths = get_all_files(data_dir)[:1]
        raw_datasets = concat_dataset(
            file_paths=file_paths,
            output_type="Dataset",
            columns="text",
            masks=["duplicate", "pattern"]
        )
        raw_datasets = raw_datasets.train_test_split(test_size=0.1, train_size=0.9)
        print()
        print("Raw Datasets loaded.")
        print(f"Train Size: {len(raw_datasets['train'])}")
        print(f"Test Size: {len(raw_datasets['test'])}")
        print()
        return raw_datasets
    
    def load_tokenizer(tokenizer_class, path_to_tokenizer):
        tokenizer = tokenizer_class.from_pretrained(path_to_tokenizer)
        print ()
        print ("Tokenizer loaded.")
        print (f"Tokenizer Class: {str(tokenizer_class)}")
        print (f"Name: {path_to_tokenizer}")
        print (f"Vocab Size: {len(tokenizer)}")
        print()
        return tokenizer
    
    def load_model(model_class, path_to_model, tokenizer):
        model = model_class.from_pretrained(path_to_model)
        model.resize_token_embeddings(len(tokenizer))
        print()
        print("Model loaded.")
        print(f"Model Class: {str(model_class)}")
        print(f"Name: {path_to_model}")
        return model
    
    def tokenize_dataset(raw_datasets, tokenizer):
        tokenize_function = lambda raw_datasets: tokenizer(raw_datasets["text"])
        tokenized_dataset = raw_datasets["train"].map(tokenize_function, batched=True)
        tokenized_dataset = tokenized_dataset.remove_columns(["text"])
        tokenized_dataset = tokenized_dataset.with_format("torch")
        return tokenized_dataset
    
    def collate_data(collator_class, tokenizer):
        data_collator = collator_class(tokenizer)
        train_dataloader = DataLoader(
            tokenized_dataset,
            batch_size=10,
            shuffle=True,
            collate_fn=data_collator
        )

        print("Data collated.")
        print("\nShape of first five batches:")
        for step, batch in enumerate(train_dataloader):
            print(batch["input_ids"].shape)
            if step > 5:
                break
        return data_collator
    

    raw_datasets = load_raw_datasets(data_dir)
    tokenizer = load_tokenizer(tokenizer_class, path_to_tokenizer)
    tokenized_dataset = tokenize_dataset(raw_datasets, tokenizer)
    model = load_model(model_class, path_to_model, tokenizer)
    data_collator = collate_data(collator_class, tokenizer)


    training_args = TrainingArguments(
        output_dir="test",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        num_train_epochs=3,
        weight_decay=0.01,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset,
        eval_dataset=raw_datasets["test"],
        tokenizer=tokenizer
    )

    trainer.train()
    trainer.save_model(output_dir)
    print(f"Language model saved to {output_dir}")
    return