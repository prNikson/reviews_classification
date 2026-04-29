import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from utils.wandb_logger import init_wandb, log_metrics, finish_wandb
from tqdm import tqdm


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
        self.save_path = cfg['models_save_path']

    def train_epoch(self) -> float:
        self.model.train()
        train_loss = 0

        progress_bar = tqdm(self.train_loader, desc='Training')

        for batch in progress_bar:
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(input_ids)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()

            progress_bar.set_postfix({"loss": train_loss / len(self.train_loader)})

        return train_loss

    def train(self):
        print(f"Starting training on {self.device}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")

        best_val_accuracy = 0

        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch + 1}/{self.num_epochs}")

            train_loss = self.train_epoch()
            val_loss, val_acc, val_f1 = self.validate()

            log_metrics({
                "epoch": epoch + 1,
                "train/loss": train_loss / len(self.train_loader),
                "val/loss": val_loss,
                "val/accuracy": val_acc,
                "val/f1_score": val_f1,
                "learning_rate": self.lr 
            })

            if val_acc > best_val_accuracy:
                best_val_accucary = val_acc
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_accuracy": val_acc,
                }, f"{self.save_path}/lstm_model_{epoch + 1}.pt")

        finish_wandb()

    def validate(self) -> tuple:
        self.model.eval()
        val_loss = 0
        preds, true_values = [], []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids)
                loss = self.criterion(outputs, labels)

                val_loss += loss.item()

                predictions = torch.argmax(outputs, dim=1)
                preds.extend(predictions.cpu().numpy())
                true_values.extend(labels.cpu().numpy())

        avg_loss = val_loss / len(self.val_loader)
        acc = accuracy_score(true_values, preds)
        f1 = f1_score(true_values, preds, average='weighted')
        
        return avg_loss, acc, f1
