from pathlib import Path

import h5py
import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset, Sampler

from helper.tokenization import load_tokenizer


class H5pyInMemoryDataset(Dataset):
    def __init__(self, name, tokenizer, split):
        data_path = (
            Path(__file__)
            .resolve()
            .parent.joinpath(f"../../configurations/{name}/tokenized_data.h5")
            .resolve()
        )
        file = h5py.File(str(data_path), "r")
        self.tokenizer = tokenizer
        self.data = file[split][:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, id):
        text = torch.tensor(self.data[id])
        label = torch.cat(
            (text[1:], torch.tensor([self.tokenizer.token_to_id("<pad>")]))
        )

        return text, label


class H5pyDataset(Dataset):
    def __init__(self, name, tokenizer, split):
        data_path = (
            Path(__file__)
            .resolve()
            .parent.joinpath(f"../../configurations/{name}/tokenized_data.h5")
            .resolve()
        )
        self.split = split
        self.file = h5py.File(str(data_path), "r")
        self.tokenizer = tokenizer

    def __len__(self):
        return self.file[self.split].shape[0]

    def __getitem__(self, ids):
        text = torch.tensor(self.file[self.split][ids])
        label = torch.cat(
            (
                text[:, 1:],
                torch.tensor([self.tokenizer.token_to_id("<pad>")])
                .repeat(len(ids))
                .unsqueeze(1),
            ),
            dim=1,
        )

        return text, label


def split_data(data, val_split=0.05):
    split_idx = int(len(data) * (1 - val_split))
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    return train_data, val_data


def create_data_loaders(name, batch_size, in_memory=True):
    tokenizer = load_tokenizer(name)

    if not in_memory:
        train_dataset = H5pyDataset(name, tokenizer, "train")
        val_dataset = H5pyDataset(name, tokenizer, "validation")

        train_dataloader = fast_loader(train_dataset, batch_size=batch_size)
        val_dataloader = fast_loader(val_dataset, batch_size=batch_size)

        return train_dataloader, val_dataloader

    train_dataset = H5pyInMemoryDataset(name, tokenizer, "train")
    val_dataset = H5pyInMemoryDataset(name, tokenizer, "validation")

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader


class SimpleBatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.batch_size = batch_size
        self.dataset_length = len(dataset)
        self.n_batches = self.dataset_length / self.batch_size
        # We always use the same order, to be able to continue within an epoch
        self.batch_ids = torch.range(0, int(self.n_batches))
        super().__init__()

    def __len__(self):
        return self.batch_size

    def __iter__(self):
        for batch_id in self.batch_ids:
            idx = torch.arange(
                batch_id * self.batch_size, (batch_id + 1) * self.batch_size
            )
            for index in idx:
                yield int(index)
        if int(self.n_batches) < self.n_batches:
            idx = torch.arange(
                int(self.n_batches) * self.batch_size, self.dataset_length
            )
            for index in idx:
                yield int(index)


def fast_loader(dataset, batch_size=32, drop_last=False):
    return DataLoader(
        dataset,
        batch_size=None,
        sampler=BatchSampler(
            SimpleBatchSampler(dataset, batch_size),
            batch_size=batch_size,
            drop_last=drop_last,
        ),
    )
