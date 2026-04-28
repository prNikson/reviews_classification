import torch
import torch.nn as nn


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, hidden_dim, num_classes, max_len=128):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.pos_embedding = nn.Embedding(max_len, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.fc = nn.Linear(embed_dim, num_classes)
        self.dropout  = nn.Dropout(0.3)

    def forward(self, input_ids, attention_mask=None):
        B, T = input_ids.shape

        positions = torch.arange(0, T).unsqueeze(0).to(input_ids.device)

        x = self.embedding(input_ids) + self.pos_embedding(positions)

        x = self.transformer(x)

        x = x.mean(dim=1)

        x = self.dropout(x)

        return self.fc(x)
