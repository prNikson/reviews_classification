import pandas as pd
from sklearn.model_selection import train_test_split

def split_dataset(cfg):
    df = pd.read_csv(cfg['train_dataset_path'])
    
    df['sentiment'] = df['sentiment'].replace({'positive': 1, 'negative': 0})

    X, y = df['review'], df['sentiment']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    return X_train.tolist(), X_test.tolist(), y_train.tolist(), y_test.tolist()
