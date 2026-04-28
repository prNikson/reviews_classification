import torch


class Config:
    # Data
    max_len = 512
    batch_size = 32

    # Model parameters
    embedding_dim = 300
    hidden_dim = 128
    dropout = 0.3

    #Training
    learning_rate = 1e-3
    num_epochs = 10
    device = "cuda" if torch.cuda.is_available() else "cpu"

    #WandB
    wandb_project="review_analysis"
