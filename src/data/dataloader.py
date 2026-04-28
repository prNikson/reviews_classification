from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from data.dataset import IMDBDataset
from data.preprocessing import split_dataset


def get_dataloaders(cfg):
    batch_size = cfg['batch_size']
    max_len = cfg['max_len']

    tokenizer = AutoTokenizer.from_pretrained(cfg["transformer_name"])

    train_texts, val_texts, train_labels, val_labels = split_dataset(cfg)

    train_dataset = IMDBDataset(train_texts, train_labels, tokenizer, max_len=max_len)
    val_dataset = IMDBDataset(val_texts, val_labels, tokenizer, max_len=max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader, tokenizer
