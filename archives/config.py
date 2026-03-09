import json

Config = {
    "working_directory": '.',
    "data_path": './data',
    "train_path": './data/train',
    "val_path": './data/val',
    "eloncam_data": '',
    "eloncam_grundtrue": '',
    "model_path": './model_checkpoints',
    "train_params": {
        "epochs": 50,
        "batch_size": 15,
        "target_size": (256, 256),
        "learning_rate": 1e-3,
    },
    "test_images_dir": './data/test/images',
    "test_masks_dir": './data/test/masks',
    "test_params": {
        "batch_size": 5,
        "target_size": (256, 256),
        "threshold": 0.5
    },
}

def update_config(updates: dict):
    """ Update the global configuration with new values 
    Args:
        updates (dict): Dictionary containing the configuration updates
    """
    Config.update(updates)

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