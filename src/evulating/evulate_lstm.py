import torch

from configs.get_config import get_config
from models.lstm_model import LSTMClassifier
from data.dataloader import get_dataloaders

def evulate(config_path: str, model_path: str, input_text: str) -> tuple[int, float]:
    cfg = get_config(config_path)
    device = cfg['device']

    _, _, tokenizer = get_dataloaders(cfg)

    encoding = tokenizer(
        input_text,
        padding="max_length",
        truncation=True,
        max_length=cfg['max_len'],
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)

    model = LSTMClassifier(tokenizer.vocab_size, cfg).to(device)
    model.load_state_dict(
        torch.load(model_path, weights_only=True)['model_state_dict']
    )
   
    model.eval()

    with torch.no_grad():
        outputs = model(input_ids)

        probs = torch.softmax(outputs, dim=1)
        predict = torch.argmax(probs, dim=1).item()
        confidence = probs[0][predict].item()

    return predict, confidence
