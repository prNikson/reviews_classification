from scripts.train_lstm import train
from evulating.evulate_lstm import evulate

if __name__ == "__main__":
#    train('configs/config.yaml')
    evulate('configs/config.yaml', 'result_models/lstm_model_20.pt')
