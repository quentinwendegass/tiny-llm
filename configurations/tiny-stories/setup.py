import sys

import h5py
from datasets import load_dataset

from src.helper.helper import load_params
from src.helper.tokenization import load_tokenizer, tokenize_and_slice, train_tokenizer

if __name__ == "__main__":
    if len(sys.argv) <= 1:
        print("Please specify mode. Available modes: 'train', 'tokenize'")
        exit(1)

    mode = sys.argv[1]

    name = "tiny-stories"

    params = load_params(name)

    if mode == "train":
        ds = load_dataset("roneneldan/TinyStories")
        data = ds["train"]["text"]
        save_path = f"configurations/{name}/tokenizer.json"
        train_tokenizer(data, params["vocab_size"], save_path)
        print(f"Tokenizer saved to {save_path}")
    elif mode == "tokenize":
        save_path = f"configurations/{name}/tokenized_data.h5"
        tokenizer = load_tokenizer(name)
        train_ds = load_dataset("roneneldan/TinyStories", split="train")
        validation_ds = load_dataset("roneneldan/TinyStories", split="validation")

        tokenized_train_texts = tokenize_and_slice(
            train_ds["text"], tokenizer, params["context_len"], params["stride"]
        )
        tokenized_val_texts = tokenize_and_slice(
            validation_ds["text"], tokenizer, params["context_len"], params["stride"]
        )

        with h5py.File(save_path, "a") as f:
            f.create_dataset("train", data=tokenized_train_texts, compression="gzip")
            f.create_dataset("validation", data=tokenized_val_texts, compression="gzip")
        print(f"Tokenized data saved to {save_path}")
    else:
        print("Unknown mode. Needs to be either 'train' or 'tokenize'")
        exit(1)
