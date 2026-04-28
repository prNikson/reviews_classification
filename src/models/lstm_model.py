import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, cfg):
        super().__init__()

        embedding_dim = cfg['embedding_dim']
        hidden_dim = cfg['hidden_dim']
        dropout = cfg['dropout']
        num_classes = cfg['num_classes']

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        _, (hidden, _) = self.lstm(x)
        out = self.dropout(hidden[-1])
        return self.fc(out)

