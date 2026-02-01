import json

Config = {
    "working_directory": '.',
    "data_path": './data',
    "train_path": './data/train',
    "val_path": './data/val',
    "eloncam_data": 'C:\\Users\\DELL\\Desktop\\Stage\\package\\data\\brute',
    "eloncam_grundtrue": 'C:\\Users\\DELL\\Desktop\\Stage\\package\\data\\grund',
    "model_path": './model_checkpoints',
    "train_params": {
        "epochs": 50,
        "batch_size": 15,
        "target_size": (256, 256),
        "learning_rate": 1e-4,
    },
}

def save_config(config: dict, path: str):
    """ Save configuration to a file 
    Args:
        config (dict): Configuration dictionary
        path (str): Path to save the configuration file
    """
    
    with open(path, 'w') as f:
        json.dump(config, f, indent=4)

def load_config(path: str) -> dict:
    """ Load configuration from a file 
    Args:
        path (str): Path to the configuration file
    Returns:   
        dict: Configuration dictionary
    """
    with open(path, 'r') as f:
        config = json.load(f)
    return config