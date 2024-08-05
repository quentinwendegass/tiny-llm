from pathlib import Path

from tokenizers import ByteLevelBPETokenizer, Tokenizer


def train_tokenizer(data, vocab_size, save_path):
    tokenizer = ByteLevelBPETokenizer()
    special_tokens = ["<pad>", "<unk>", "<s>", "</s>", "<mask>"]
    tokenizer.train_from_iterator(
        data, vocab_size=vocab_size, special_tokens=special_tokens
    )
    tokenizer.enable_padding(pad_id=tokenizer.token_to_id("<pad>"), pad_token="<pad>")

    tokenizer.save(save_path)
    return tokenizer


def load_tokenizer(name):
    tokenizer_path = (
        Path(__file__)
        .resolve()
        .parent.joinpath(f"../../configurations/{name}/tokenizer.json")
        .resolve()
    )
    return Tokenizer.from_file(str(tokenizer_path))


def tokenize_and_slice(texts, tokenizer, max_length, stride):
    tokenized_texts = []
    for text in texts:
        tokens = []
        if len(text) > 2000:  # arbitrary number that hopefully fits into the tokenizer
            for i in range(0, len(text), 2000):
                text_chunk = text[i : i + 2000]
                tokens.append(tokenizer.encode(text_chunk).ids)
        else:
            tokens.append(tokenizer.encode(text).ids)

        tokens = [item for sublist in tokens for item in sublist]
        start = 0
        while start < len(tokens):
            end = start + max_length
            chunk = tokens[start:end]
            if len(chunk) < max_length:
                chunk = chunk + [tokenizer.token_to_id("<pad>")] * (
                    max_length - len(chunk)
                )
            tokenized_texts.append(chunk)
            start += stride

    return tokenized_texts
