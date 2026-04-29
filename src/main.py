import argparse

parser = argparse.ArgumentParser()

parser.add_argument("action", help="train, evulate or compare")
parser.add_argument("--model_arch", help="lstm, bert or None")
parser.add_argument("--config_file")
parser.add_argument("--model_path")

args = parser.parse_args()


if args.action == "compare":
    from evulating.compare_models import compare_models
    compare_models("configs/config.yaml")

if args.model_arch == "lstm":
    from scripts.train_lstm import train
    from evulating.evulate_lstm import evulate
    
    if args.action == "train":
        train(args.config_file)
    elif args.action == "evulate":
        evulate(args.config_file, args.model_path)
    else:
        print("Undefined value")

elif args.model_arch == "bert":
    from scripts.train_bert import train
    from evulating.evulate_bert import evulate

    if args.action == "train":
        train(args.config_file)
