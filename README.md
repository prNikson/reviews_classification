# Sentiment Analysis Comparison: LSTM vs BiLSTM vs DistilBERT vs BERT

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/)

**Сравнительный анализ трех архитектур для задачи классификации тональности текста**

</div>

---

## Описание проекта

Проект посвящен сравнению трех архитектур нейронных сетей для задачи бинарной классификации тональности текстовых отзывов:

| Модель | Архитектура | Особенности |
|--------|-------------|-------------|
| **LSTM** | LSTM + Dropout | Классический RNN подход с механизмом долгой краткосрочной памяти |
| **BiLSTM** | Bidirectional LSTM + Dropout | Двунаправленная LSTM |
| **DistilBERT** | Lightweight Transformer | Дистиллированная версия BERT (на 40% легче) |
| **BERT-base** | Full Transformer | Полноценная трансформерная архитектура от Google |

### Цель исследования

1. Сравнить качество классификации (Accuracy, F1, Precision, Recall)
2. Измерить производительность (время инференса)
3. Оценить компромиссы между качеством и скоростью

### Используемый датасет
https://huggingface.co/datasets/prNikson/imdb_dataset/tree/main

- imdb_dataset.csv - классический датасет отзывов с IMDB на 50К отзывов
- hybrid_dataset.csv - датасет для тестирования на 1000 отзывов с соотношением 70/30 синтетических и реалистичных отзывов

## Обучение моделей
### Экспериментальная установка

| Параметр | LSTM | BiLSTM |DistilBERT | BERT |
|----------|------|------|------------|------|
| Эпохи | 5 | 5 | 3 | 3 |
| Batch size | 32 | 32 | 16 | 16 |
| Learning rate | 1e-3 |1e-3|  2e-5 | 2e-5 |
| Optimizer | Adam | Adam | AdamW | AdamW |
| Max length | 512 | 512 | 512 | 512 |
| GPU | RTX A5000 (24 GB) | RTX A5000 | RTX A5000 | RTX A5000 |

## Результаты тестирования моделей

### Метрики качества

<div align="center">

| Модель      | Accuracy | F1-Score | Precision | Recall | Inference (сек)<br>на 1к примерах | Параметры |
|-------------|----------|----------|-----------|--------|-----------------|-----------|
| LSTM        | 86.60%   | 86.52%   | 87.51%    | 86.60% | **1.17**        |   7.5M    |
| BiLSTM      | 87.90%   | 87.87%   | 88.24%    | 87.90% | 1.28            |   7.76M   |
| DistilBERT  | 91.10%   | 91.07%   | 91.64%    | 91.10% | 5.01            |   66M     |
| BERT        |**97.50%**|**97.50%**|**97.52%** |**97.50%**| 8.96          |   110M    |

</div>

### Визуализация Accuracy
████████████████████████████████████████ 97.5% BERT
█████████████████████████████████░░░░░░░ 91.1% DistilBERT
█████████████████████████████░░░░░░░░░░░ 87.9% BiLSTM
████████████████████████████░░░░░░░░░░░░ 86.6% LSTM

### Результаты

- **Лучшее качество**: BERT (97.5% accuracy) — превосходит LSTM на **+10.9%**
- **Лучшая скорость**: LSTM (1.17 сек) — быстрее BERT в **4.1 раза**
- **Лучший баланс**: DistilBERT — 91.1% quality при 2x скорости vs BERT

## Запуск обучения, инференса и сравнения моделей
usage: `main.py [-h] [--model_arch MODEL_ARCH] [--config_file CONFIG_FILE] [--model_path MODEL_PATH] action`<br>
train: `uv run main.py train --model_arch lstm (bert) --config_file configs/config.yaml`<br>
evulate: `uv run main.py evulate --model_arch lstm (bert) --config_file configs/config.yaml --model_path models/model.pt`<br>
compare: `uv run main.py compare`
