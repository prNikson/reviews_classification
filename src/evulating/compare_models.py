import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
        BertForSequenceClassification,
        DistilBertForSequenceClassification,
        AutoTokenizer)

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, classification_report, confusion_matrix
)

import pandas as pd
from sklearn.calibration import calibration_curve
from tqdm import tqdm

import time

from data.dataset import IMDBDataset
from configs.get_config import get_config
from models.lstm_model import LSTMClassifier

def read_dataset(test_dataset_path="dataset/hybrid_dataset.csv"):
    df = pd.read_csv(test_dataset_path)

    texts = df['review'].tolist()
    labels = df['sentiment'].tolist()
    
    return texts, labels

def test_lstm(cfg):
    texts, labels = read_dataset()

    tokenizer = AutoTokenizer.from_pretrained(cfg['transformer_name'])
    max_len = cfg['max_len']
    batch_size = 16
    device = 'cuda'

    test_dataset = IMDBDataset(texts, labels, tokenizer, max_len=max_len)    
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    all_preds = []
    all_labels = []
    all_probs = []
    
    start_time = time.time()

    model = LSTMClassifier(tokenizer.vocab_size, cfg, bidirectional=cfg['bidirectional']).to(device)
    model.load_state_dict(torch.load(cfg['models_save_path'] + '/lstm_model_20.pt')['model_state_dict'])

    model.eval()

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evulating lstm"):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids)

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    inference_time = time.time() - start_time


    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')

    positive_probs = [p[1] for p in all_probs]

    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  Inference: {inference_time:.2f} sec")

def test_bert(cfg, type_model='distil'):
    texts, labels = read_dataset()    

    tokenizer = AutoTokenizer.from_pretrained(cfg['transformer_name'])
    max_len = cfg['max_len']
    batch_size = 12
    device = 'cuda'

    test_dataset = IMDBDataset(texts, labels, tokenizer, max_len=max_len)    
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    all_preds = []
    all_labels = []
    all_probs = []
    
    if type_model == "distil":
        model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels=cfg['num_classes']
        )
        model_path = "/bert_model_4.pt"
    else:
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=cfg['num_classes']
        )
        model_path = "/bert_model_3.pt"

    model.load_state_dict(torch.load(cfg['models_save_path'] + model_path)['model_state_dict'])

    start_time = time.time()

    model.eval()
    model.to(device)

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evulating bert"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)

            probs = torch.softmax(outputs.logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    inference_time = time.time() - start_time


    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')

    positive_probs = [p[1] for p in all_probs]

    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  Inference: {inference_time:.2f} sec")

def compare_models(config_path):
    cfg = get_config(config_path)

    test_lstm(cfg)
    test_bert(cfg, type_model='distil')
    test_bert(cfg, type_model='base')


