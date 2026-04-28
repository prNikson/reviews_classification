import wandb
from training.config import Config

def init_wandb(cfg, model):
    wandb.init(
        project=cfg['wandb_project'],
        config=cfg
    )

    return wandb.run

def log_metrics(metrics_dict: dict, step=None):
    wandb.log(metrics_dict, step=step)

def finish_wandb():
    if wandb.run:
        wandb.finish()
