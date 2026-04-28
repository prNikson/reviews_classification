import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from utils.wandb_logger import init_wandb, log_metrics, finish_wandb


class Trainer:
    def __init__(self, model, train_loader, val_loader, cfg):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self._load_config(cfg)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()

        init_wandb(cfg, self.model)

    def _load_config(self, cfg):
        self.device = cfg['device']
        self.lr = float(cfg['learning_rate'])
        self.num_epochs = cfg['num_epochs']

    def train_epoch(self) -> float:
        self.model.train()
        train_loss = 0

        for batch in self.train_loader:
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(input_ids)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()

        return train_loss

    def train(self):
        for epoch in range(self.num_epochs):
            train_loss = self.train_epoch()
            self.validate(epoch, train_loss)

    def validate(self, epoch, train_loss):
        self.model.eval()
        val_loss = 0
        preds, true_values = [], []

        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids)
                loss = self.criterion(outputs, labels)

                val_loss += loss.item()

                predictions = torch.argmax(outputs, dim=1)
                preds.extend(predictions.cpu().numpy())
                true_values.extend(labels.cpu().numpy())

        acc = accuracy_score(true_values, preds)
        f1 = f1_score(true_values, preds, average='weighted')
        
        log_metrics({
            "epoch": epoch + 1,
            "train/loss": train_loss,
            "val/loss": val_loss,
            "val/accuracy": acc,
            "val/f1_score": f1
        })
