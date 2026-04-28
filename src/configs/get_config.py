import yaml

def get_config(config_file='configs/config.yaml') -> dict:
    with open(config_file, 'r') as file:
        cfg = yaml.safe_load(file)

    return cfg 
