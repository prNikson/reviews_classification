from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import torch

from configs.get_config import get_config
from data.dataloader import get_dataloaders
from training.trainer_bert import Trainer


def train(config_path='configs/config_bert.yaml'):
    cfg = get_config(config_path)

    train_loader, val_loader, tokenizer = get_dataloaders(cfg)

    model = DistilBertForSequenceClassification.from_pretrained(
        cfg['transformer_name'],
        num_labels=cfg['num_classes']
    ).to(cfg['device'])

    trainer = Trainer(model, train_loader, val_loader, cfg)
    trainer.train()
