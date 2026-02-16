
import logging
from config import Config, save_config, load_config
import os, pathlib 
from dataset.load_eloncam_data import *
from metrics import dice_coeff, iou_score, mse, rmse, hausdorff_distance
from train_model import run_full_train
from model_test import run_prediction
from plot_loss import plot_loss, load_loss_history
from train_model_map import run_map_train


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
    data_path = pathlib.Path(Config.get("dataset_src", src.joinpath('dataset')))
  
    train_path = data_path.joinpath('train')
    val_path = data_path.joinpath('val')
    test_path = data_path.joinpath('test')
    Config["test_images_dir"] = str(test_path.joinpath('images'))
    Config["test_masks_dir"] = str(test_path.joinpath('masks'))
    Config["train_path"] = str(train_path)
    Config["val_path"] = str(val_path)
    Config["test_path"] = str(test_path)
    Config["model_path"] = str(src.joinpath('checkpoints'))
    Config["data_path"] = str(data_path)

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
    check_working_directory(ws_path=ws_path)
    

    if kwargs.get("dataset_src", None) is not None:
        Config["dataset_src"] = kwargs.get("dataset_src")

    exist_data = get_existing_data(dest_path= kwargs.get("dataset_src", ws_path + '/dataset'))
    if exist_data:
        if pathlib.Path(ws_path).joinpath('config.json').exists():
            print("Existing dataset found. Skipping data loading.")
            Config.update(load_config(os.path.join(ws_path, 'config.json')))
        else:
            save_config(Config, os.path.join(ws_path, 'config.json'))
    else:
        logging.error("No existing dataset found. Please load data before training.")
        return
    
    if Config["train_params"]["mode"] == "regression":
        print("Starting training with regression mode ...")
        run_map_train(root=Config.get("data_path", './data'),
                   epochs=Config.get("train_params", {}).get("epochs", 50),
                   batch_size=Config.get("train_params", {}).get("batch_size", 20),
                   train_dir=Config.get("train_path", './data/train'),
                   val_dir=Config.get("val_path", './data/val'),
                   target_size=Config.get("train_params", {}).get("target_size", (256, 256)),
                    lr=Config.get("train_params", {}).get("learning_rate", 1e-4),
                   save_dir=Config.get("model_path", './model_checkpoints'),
                   history_path=os.path.join(ws_path, 'training_history.json'))
        
        history = load_loss_history(os.path.join(ws_path, 'training_history.json'))
        plot_loss(history[0], history[1],
              out_path=os.path.join(ws_path, 'loss_plot.png'),
                metric_name='SmoothL1 Loss')
        logging.info("Running prediction and evaluation...")
        run_prediction(
            model_path=os.path.join(Config.get("model_path", './model_checkpoints'), 'unet_last_epoch.pth'),
            images_dir=Config.get("test_images_dir", './data/test/images'),
            out_masks_dir=os.path.join(ws_path, 'test_output2'),
            target_size=tuple(Config.get("test_params", {}).get("target_size", (256, 256))),
            threshold=Config.get("test_params", {}).get("threshold", 0.5),
            history_path=os.path.join(ws_path, 'last_epoch_prediction_history.json'),
            metrics={
                'rmse': rmse,
                'hausdorff': hausdorff_distance
            }
        )
    
    else:
        print("Starting training ...")
        run_full_train(root=Config.get("data_path", './data'),
                    epochs=Config.get("train_params", {}).get("epochs", 50),
                    batch_size=Config.get("train_params", {}).get("batch_size", 20),
                    train_dir=Config.get("train_path", './data/train'),
                    val_dir=Config.get("val_path", './data/val'),
                    target_size=Config.get("train_params", {}).get("target_size", (256, 256)),
                        lr=Config.get("train_params", {}).get("learning_rate", 1e-4),
                    save_dir=Config.get("model_path", './model_checkpoints'),
                    history_path=os.path.join(ws_path, 'training_history.json'))

        logging.info("Plotting training history...")
        history = load_loss_history(os.path.join(ws_path, 'training_history.json'))
        plot_loss(history[0], history[1],
                out_path=os.path.join(ws_path, 'loss_plot.png'),
                    metric_name='dice')
        

        logging.info("Running prediction and evaluation...")
        
        run_prediction(
            model_path=os.path.join(Config.get("model_path", './model_checkpoints'), 'unet_last_epoch.pth'),
            images_dir=Config.get("test_images_dir", './data/test/images'),
            out_masks_dir=os.path.join(ws_path, 'test_output2'),
            target_size=tuple(Config.get("test_params", {}).get("target_size", (256, 256))),
            threshold=Config.get("test_params", {}).get("threshold", 0.5),
            history_path=os.path.join(ws_path, 'last_epoch_prediction_history.json'),
            metrics={
                'dice': dice_coeff,
                'iou': iou_score
            }
        )

    logging.info("All tasks completed.")



if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    parseur = argparse.ArgumentParser(description="Run training tasks.")
    parseur.add_argument('--ws_path', type=str, default='.',
                        help='Path to working directory')
    parseur.add_argument('--eloncam_data', type=str, default='',
                        help='Path to Eloncam data directory')
    parseur.add_argument('--eloncam_grundtrue', type=str, default='',
                        help='Path to Eloncam groundtruth directory')
    
    parseur.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parseur.add_argument('--batch_size', type=int, default=15,
                        help='Training batch size')
    parseur.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate for training')
    
    parseur.add_argument('--dataset_src', type=str, default="./dataset",
                        help='Path to existing dataset (if any)')
    
    parseur.add_argument('--mode', type=str, default="binary", choices=["binary", "regression"],
                        help='Path to existing dataset (if any)')
    
    args = parseur.parse_args()
    if args.eloncam_data != '':
        Config["eloncam_data"] = args.eloncam_data
    if args.eloncam_grundtrue != '':
        Config["eloncam_grundtrue"] = args.eloncam_grundtrue
    Config["train_params"]["epochs"] = args.epochs
    Config["train_params"]["batch_size"] = args.batch_size
    Config["train_params"]["learning_rate"] = args.learning_rate
    Config["train_params"]["mode"] = args.mode

    run_tasks(ws_path=args.ws_path, dataset_src=args.dataset_src,)