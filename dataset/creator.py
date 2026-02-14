
import pathlib
import shutil
import logging
from load_eloncam_data import *

class DatasetCreator:
    def __init__(self, dataset_path: str, data_path: str,
                 annotation_path: str,
                 image_format: str = "tiff",
                 split_ratio: tuple = (0.6, 0.25, 0.15),
                 **kwargs):
        
        self.dataset_path = dataset_path
        self.data_path = data_path
        self.annotation_path = annotation_path
        self.image_format = image_format
        self.split_ratio = split_ratio
        self.annotation_path_dict = {}

        self._datas = []
        self.train_data = []        
        self.val_data = []
        self.test_data = []

        self.validate_data(exclude_dir=kwargs.get("exclude_dir", []))


    def get_serial_data(self, pth: str = None, exclude_dir: list = []):
        path = pathlib.Path(pth)
        folders = []    
        fd = [f for f in path.iterdir() if f.is_dir() and f.name not in exclude_dir]
        while len(fd) > 0:
            fs = fd.pop(0)
            f = str(fs.name)
            if f.split('_')[0].isnumeric() and str(f).split('_')[-1].isnumeric():
                folders.append(fs)
            else:
                fd.extend([f for f in fs.iterdir() if f.is_dir() and f.name not in exclude_dir])

        return folders
    
    def validate_data(self, exclude_dir: list = []):
        brut_data = self.get_serial_data(pth=self.data_path)
        analysed_data = self.get_serial_data(pth=self.annotation_path, exclude_dir=exclude_dir)
        for d in brut_data:
            for an in [f for f in analysed_data]:
                if d.name == an.name:
                    self.annotation_path_dict[d] = pathlib.Path(self.annotation_path) / an
                    self._datas.append(d)
                    break
                else:
                    logging.warning(f"No corresponding annotation found for {d}")
        
        logging.info(f"Found {len(self._datas)} valid data folders with corresponding annotations.")
                
    
    def split_data(self, data: list = None):
        if data is None:
            data = self._datas

        n = len(data)
        n_train = int(n * self.split_ratio[0])
        n_val = int(n * self.split_ratio[1])
        self.train_data = data[:n_train]
        self.val_data = data[n_train:n_train+n_val]
        self.test_data = data[n_train+n_val:]

        return self.train_data, self.val_data, self.test_data
    
    def create_no_serial_dataset(self, n_serial: int = 10):
        # This method can be implemented to create a dataset without relying on serial naming conventions
        _data = self._datas[:n_serial]
        train_data, val_data, test_data = self.split_data(data=_data)
        
        pathlib.Path(self.dataset_path).mkdir(parents=True, exist_ok=True)
        train_pt = pathlib.Path(self.dataset_path).joinpath("train")
        train_pt.mkdir(parents=True, exist_ok=True)
        train_pt.joinpath("images").mkdir(parents=True, exist_ok=True)
        train_pt.joinpath("masks").mkdir(parents=True, exist_ok=True)
        
        val_pt = pathlib.Path(self.dataset_path).joinpath("val")
        val_pt.mkdir(parents=True, exist_ok=True)
        val_pt.joinpath("images").mkdir(parents=True, exist_ok=True)
        val_pt.joinpath("masks").mkdir(parents=True, exist_ok=True)

        test_pt = pathlib.Path(self.dataset_path).joinpath("test")
        test_pt.mkdir(parents=True, exist_ok=True)
        test_pt.joinpath("images").mkdir(parents=True, exist_ok=True)
        test_pt.joinpath("masks").mkdir(parents=True, exist_ok=True)

        for path in  train_data:
            train_data = [str(p) for p in path.iterdir() if p.is_file() and p.suffix == f".{self.image_format}"]
            groundtrue_files = [f.name for f in pathlib.Path(self.annotation_path_dict[path]).iterdir() if f.is_file() and f.suffix == f".{self.image_format}"]
            
            save_dataset(train_data, self.dataset_path, 
                 groundtrue_files, 
                 self.annotation_path_dict[path],
                 mask = 'hsv',
                 data_type = 'train')
        
        for path in  val_data:
            val_data = [str(p) for p in path.iterdir() if p.is_file() and p.suffix == f".{self.image_format}"]
            groundtrue_files = [f.name for f in pathlib.Path(self.annotation_path_dict[path]).iterdir() if f.is_file() and f.suffix == f".{self.image_format}"]
            
            save_dataset(val_data, self.dataset_path, 
                 groundtrue_files, 
                 self.annotation_path_dict[path],
                 mask = 'hsv',
                 data_type = 'val')
            
        for path in  test_data:
            test_data = [str(p) for p in path.iterdir() if p.is_file() and p.suffix == f".{self.image_format}"]
            groundtrue_files = [f.name for f in pathlib.Path(self.annotation_path_dict[path]).iterdir() if f.is_file() and f.suffix == f".{self.image_format}"]
            
            save_dataset(test_data, self.dataset_path, 
                 groundtrue_files, 
                 self.annotation_path_dict[path],
                 mask = 'hsv',
                 data_type = 'test')
        
    def crate_serial_dataset(self, n_serial: int = 10):
        _data = self._datas[:n_serial]
        train_data, val_data, test_data = self.split_data(data=_data)
        
        pathlib.Path(self.dataset_path).mkdir(parents=True, exist_ok=True)
        train_pt = pathlib.Path(self.dataset_path).joinpath("train")
        train_pt.mkdir(parents=True, exist_ok=True)
        train_pt.joinpath("images").mkdir(parents=True, exist_ok=True)
        train_pt.joinpath("masks").mkdir(parents=True, exist_ok=True)
        
        val_pt = pathlib.Path(self.dataset_path).joinpath("val")
        val_pt.mkdir(parents=True, exist_ok=True)
        val_pt.joinpath("images").mkdir(parents=True, exist_ok=True)
        val_pt.joinpath("masks").mkdir(parents=True, exist_ok=True)

        test_pt = pathlib.Path(self.dataset_path).joinpath("test")
        test_pt.mkdir(parents=True, exist_ok=True)
        test_pt.joinpath("images").mkdir(parents=True, exist_ok=True)
        test_pt.joinpath("masks").mkdir(parents=True, exist_ok=True)

       
if __name__ == "__main__":
    dataset_path = "F:\Data_Eloncam\Dataset"  # This is where the final dataset will be created
    data_path = "F:\Data_Eloncam\Data"
    annotation_path = "F:\Data_Eloncam\Analysed_data"

    creator = DatasetCreator(dataset_path, data_path, annotation_path, 
                             exclude_dir=['outputs'])  # Exclude specific serials if needed
    creator.create_no_serial_dataset(n_serial=10)  # or creator.create_no_serial_dataset(n_serial=10) for a non-serial dataset
    print(f"Dataset created successfully at {dataset_path}")


    