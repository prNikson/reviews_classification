import torch
import torch.nn as nn
import torch.optim as optim
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score
from utils.wandb_logger import init_wandb, log_metrics, finish_wandb
from tqdm import tqdm


class Trainer:
    def __init__(self, model, train_loader, val_loader, cfg):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self._load_config(cfg)

        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.lr,
            weight_decay=cfg.get('weight_decay', 0.01)
        )
        
        total_steps = len(train_loader) * self.num_epochs
        warmup_steps = int(cfg.get('warmup_ratio', 0.1) * total_steps)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        self.criterion = nn.CrossEntropyLoss()
        
        init_wandb(cfg, self.model)

    def _load_config(self, cfg):
        self.device = cfg['device']
        self.lr = float(cfg['learning_rate'])
        self.num_epochs = cfg['num_epochs']
        self.gradient_clip = cfg.get('gradient_clip', 1.0)
        self.save_path = cfg['models_save_path']

    def train_epoch(self) -> tuple:
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        progress_bar = tqdm(self.train_loader, desc='Training')
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            logits = outputs.logits
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            progress_bar.set_postfix({'loss': loss.item()})
        
        train_acc = accuracy_score(all_labels, all_preds)
        avg_loss = total_loss / len(self.train_loader)
        
        return avg_loss, train_acc

    def train(self):
        print(f"Starting training on {self.device}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        
        best_val_accuracy = 0
        
        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch + 1}/{self.num_epochs}")
            
            train_loss, train_acc = self.train_epoch()
            
            val_loss, val_acc, val_f1 = self.validate()
            
            log_metrics({
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "train/accuracy": train_acc,
                "val/loss": val_loss,
                "val/accuracy": val_acc,
                "val/f1_score": val_f1,
                "learning_rate": self.scheduler.get_last_lr()[0]
            })
            
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_accuracy': val_acc,
                }, f"{self.save_path}/bert_model_{epoch + 1}.pt")

        finish_wandb()

    def validate(self) -> tuple:
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                logits = outputs.logits
                
                total_loss += loss.item()
                
                predictions = torch.argmax(logits, dim=1)
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        return avg_loss, accuracy, f1
