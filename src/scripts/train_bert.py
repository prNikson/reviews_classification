from transformers import DistilBertForSequenceClassification, BertForSequenceClassification
import torch

from configs.get_config import get_config
from data.dataloader import get_dataloaders
from training.trainer_bert import Trainer


def train(config_path='configs/config_bert.yaml'):
    cfg = get_config(config_path)

    train_loader, val_loader, tokenizer = get_dataloaders(cfg)

    if cfg['transformer_name'].startswith("bert"):
        bert_model = "bert-base-uncased"
        model = BertForSequenceClassification.from_pretrained(
            bert_model,
            num_labels=cfg.get('num_classes', 2)
        ).to(cfg['device'])
    
    else:
        distilbert_model = "distilbert-base-uncased"
        model = DistilBertForSequenceClassification.from_pretrained(
            distilbert_model,
            num_labels=cfg.get('num_classes', 2)
        ).to(cfg['device'])

    trainer = Trainer(model, train_loader, val_loader, cfg)
    trainer.train()
