
from config import Config, save_config, load_config
import os, pathlib 
from dataset.load_eloncam_data import *
from train_model import run_full_train
from plot_loss import plot_loss, load_loss_history

def check_working_directory(ws_path: str):
    """ Check and create working directory structure 
    Args:
        ws_path (str): Path to working directory
    """

    working_directory = Config.get("working_directory", '.')
    if working_directory != ws_path:
        Config["working_directory"] = ws_path
        working_directory = ws_path
    if not os.path.exists(working_directory):
        os.makedirs(working_directory)
    
    src = pathlib.Path(working_directory)
    src.joinpath('checkpoints').mkdir(parents=True, exist_ok=True)
    #src.joinpath('logs').mkdir(parents=True, exist_ok=True)
    data_path = src.joinpath('dataset') 
    data_path.mkdir(parents=True, exist_ok=True)
    Config["data_path"] = str(data_path)
    train_path = data_path.joinpath('train')
    val_path = data_path.joinpath('val')
    train_path.mkdir(parents=True, exist_ok=True)
    val_path.mkdir(parents=True, exist_ok=True)
    train_path.joinpath('images').mkdir(parents=True, exist_ok=True)
    train_path.joinpath('masks').mkdir(parents=True, exist_ok=True)
    val_path.joinpath('images').mkdir(parents=True, exist_ok=True)
    val_path.joinpath('masks').mkdir(parents=True, exist_ok=True)
    Config["train_path"] = str(train_path)
    Config["val_path"] = str(val_path)
    Config["model_path"] = str(src.joinpath('checkpoints'))

def get_existing_data(dest_path: str):
    """ Check for existing data in the specified directory 
    Args:
        dest_path (str): Path to data directory
    Returns:
        bool: True if data exists, False otherwise
    """
    data_path = pathlib.Path(dest_path)
    train_images = data_path.joinpath('train/images')
    val_images = data_path.joinpath('val/images')
    if train_images.exists() and val_images.exists():
        if any(train_images.iterdir()) and any(val_images.iterdir()):
            return True
    return False

def load_data(eloncam_data: str, eloncam_grundtrue: str,
              **kwargs):
    """ Load data from specified directory 
    Args:
        eloncam_data (str): Path to eloncam data directory
        eloncam_grundtrue (str): Path to eloncam groundtruth directory
    """
    des = kwargs.get("dest_path", Config.get("data_path", './data'))
    
    create_dataset(dest_path=des,
                image_path=eloncam_data,
                grundtruth_path=eloncam_grundtrue,
                ext = ".tiff",
                val_size=0.4)

    
def run_tasks(**kwargs):
    """ Run tasks with specified working directory 
    Args:
        ws_path (str): Path to working directory
    """
    print("checking working directory...")

    ws_path = kwargs.get("ws_path", '.')
    exist_data = get_existing_data(dest_path=os.path.join(ws_path, 'dataset'))
    if exist_data:
        print("Existing dataset found. Skipping data loading.")
        Config = load_config(os.path.join(ws_path, 'config.json'))
    else:
        check_working_directory(ws_path)

        print("Loading Eloncam dataset...")
        save_config(Config.copy(), os.path.join(ws_path, 'config.json'))

        eloncam_data = Config.get("eloncam_data", '')
        eloncam_grundtrue = Config.get("eloncam_grundtrue", '')
        if eloncam_data != '' and eloncam_grundtrue != '':
            load_data(eloncam_data, eloncam_grundtrue,
                    dest_path=Config.get("data_path", './data'))
    
    print("Starting training...")
    run_full_train(root=Config.get("data_path", './data'),
                   epochs=Config.get("train_params", {}).get("epochs", 50),
                   batch_size=Config.get("train_params", {}).get("batch_size", 20),
                   train_dir=Config.get("train_path", './data/train'),
                   val_dir=Config.get("val_path", './data/val'),
                   target_size=Config.get("train_params", {}).get("target_size", (256, 256)),
                   save_dir=Config.get("model_path", './model_checkpoints'),
                   history_path=os.path.join(ws_path, 'training_history.json'))

    print("Plotting training history...")
    history = load_loss_history(os.path.join(ws_path, 'training_history.json'))
    plot_loss(history[0], history[1],
              out_path=os.path.join(ws_path, 'loss_plot.png'),
                metric_name='dice')
    
    print("All tasks completed.")



if __name__ == "__main__":
    import argparse
    
    parseur = argparse.ArgumentParser(description="Run training tasks.")
    parseur.add_argument('--ws_path', type=str, default='.',
                        help='Path to working directory')
    parseur.add_argument('--eloncam_data', type=str, default='',
                        help='Path to Eloncam data directory')
    parseur.add_argument('--eloncam_grundtrue', type=str, default='',
                        help='Path to Eloncam groundtruth directory')
    
    parseur.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parseur.add_argument('--batch_size', type=int, default=15,
                        help='Training batch size')
    
    
    args = parseur.parse_args()
    if args.eloncam_data != '':
        Config["eloncam_data"] = args.eloncam_data
    if args.eloncam_grundtrue != '':
        Config["eloncam_grundtrue"] = args.eloncam_grundtrue
    Config["train_params"]["epochs"] = args.epochs
    Config["train_params"]["batch_size"] = args.batch_size

    run_tasks(ws_path=args.ws_path)