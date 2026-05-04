import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, cfg, bidirectional=True):
        super().__init__()

        embedding_dim = cfg['embedding_dim']
        hidden_dim = cfg['hidden_dim']
        dropout = cfg['dropout']
        num_classes = cfg['num_classes']

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(fc_input_dim, num_classes)

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        _, (hidden, _) = self.lstm(x)
        
        if self.lstm.bidirectional:
            forward_hidden = hidden[-2, :, :]
            backward_hidden = hidden[-1, :, :]

            final_hidden = torch.cat((forward_hidden, backward_hidden), dim=1)
        else:
            final_hidden = hidden[-1, :, :]

        out = self.dropout(final_hidden)
        return self.fc(out)

