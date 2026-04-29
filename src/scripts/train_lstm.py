from configs.get_config import get_config
from models.lstm_model import LSTMClassifier
from data.dataloader import get_dataloaders
from training.trainer import Trainer

def train(config_path='configs/config.yaml'):
    cfg = get_config(config_path)

    train_loader, val_loader, tokenizer = get_dataloaders(cfg)

    model = LSTMClassifier(tokenizer.vocab_size, cfg).to(cfg['device'])

    trainer = Trainer(model, train_loader, val_loader, cfg)
    trainer.train()
