import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
import wandb

from dataloader import get_train_test_loader
from models.lstm_model import LSTMClassifier
from models.transformer_model import TransformerClassifier


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, test_loader, tokenizer = get_train_test_loader()

model = TransformerClassifier(
    vocab_size = tokenizer.vocab_size,
    embed_dim=128,
    hidden_dim=128,
    num_heads=4,
    num_classes=2
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

criterion = nn.CrossEntropyLoss()

best_test_loss = float('inf')
patience = 3
counter = 0
num_epochs = 10

run = wandb.init(project='review_analysis', config={"num_epochs": 10, "learning_rate": 1e-3})

for epoch in range(num_epochs):
    model.train()
    train_loss = 0

    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    model.eval()
    test_loss = 0
    preds, true = [], []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids)
            loss = criterion(outputs, labels)

            test_loss += loss.item()

            predictions = torch.argmax(outputs, dim=1)
            preds.extend(predictions.cpu().numpy())
            true.extend(labels.cpu().numpy())

    acc = accuracy_score(true, preds)
    f1 = f1_score(true, preds, average="weighted")

    print("Epoch:", epoch)

    wandb.log({
        "train_loss": train_loss,
        "test_loss": test_loss,
        "accuracy": acc,
        "f1_score": f1
    })
