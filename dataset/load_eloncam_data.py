
import os
import pathlib
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from mat import distance_map

def load_images_path(path:str, ext:str = "tiff"):
  """
  Load all images path from a directory
  :param path: path to the directory
  :param ext: extension of the images
  :return: list of images path
  """
  P = pathlib.Path(path)
  images_path = list(P.glob(f"*.{ext}"))
  return images_path

def crop_images(path:str, isdir:bool = False, output_size:tuple = (512,512), ext:str = "tiff", lenght = None):

  def crop(image):
      start_row = 100
      end_row = image.shape[0] - start_row

      start_col = 250
      end_col = image.shape[1] - start_col

      # Crop the image

      crop_image = image[start_row:end_row, start_col:end_col]
      crop_image = cv2.resize(crop_image, output_size)
      return crop_image

  if isdir:
    paths = load_images_path(path, ext)
    crop_images = []
    if lenght is not None:
      paths = paths[:lenght]
    for path in paths:
      image = cv2.imread(path)
      crop_images.append(crop(image))
    return crop_images
  else:
    image = cv2.imread(path)
    return crop(image)

def get_mask_from_hsv(image:np.ndarray) :
  hsv = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2HSV)
  s_im = hsv[:,:,1]
  return s_im 

def get_mask_from_map(image:np.ndarray) :
  mask = distance_map(image)
  return mask

def decouper_image(image, n, image_size = None):
    """
    Découpe une image en n x n imagettes.

    Paramètres
    ----------
    image : np.ndarray ou str
        L'image à découper (chemin ou tableau).
    n : int
        Nombre de divisions par dimension (n x n imagettes).

    Retour
    ------
    list[np.ndarray]
        Liste contenant les imagettes (du haut-gauche au bas-droit).
    """

    # Si l'entrée est un chemin de fichier, on charge l'image
    if isinstance(image, str):
        image = cv2.imread(image)
        if image_size:
          image = cv2.resize(image, image_size)
        if image is None:
            raise ValueError("Impossible de charger l'image depuis le chemin fourni.")

    h, w, _ = np.array(image).shape

    # Dimensions des imagettes
    h_tile = h // n
    w_tile = w // n

    imagettes = []

    for i in range(n):
        for j in range(n):
            y1, y2 = i * h_tile, (i + 1) * h_tile
            x1, x2 = j * w_tile, (j + 1) * w_tile
            imagette = image[y1:y2, x1:x2]
            imagettes.append(imagette)

    return imagettes 

def load_images(path:str, extention = '.tiff'):
  images = []
  if not os.path.isdir(path):
    image = cv2.imread(path)
    return image.astype(np.float32)
  else:
    for file in os.listdir(path):
      if file.endswith(extention):
        image = cv2.imread(os.path.join(path, file))
        images.append(image.astype(np.float32))
    return images

def get_image_paths(image_path: str, grundtruth_path: str = None, ext: str = ".tiff"):
  image_paths = []
  gimage_paths = []
  missing_gimages = []

  if grundtruth_path is not None:
    for file in os.listdir(grundtruth_path):
      if file.endswith(ext):
        gimage_paths.append(file)

  for file in os.listdir(image_path):
    if file.endswith(ext) and file in gimage_paths:
      image_paths.append(os.path.join(image_path, file))
    else:
      missing_gimages.append(file)
  return image_paths,missing_gimages

def save_dataset(image_paths: list, dest_image_path: str,groundtrue_files: list, 
                 groundtrue_src: str, 
                 mask = 'hsv',
                 data_type = 'train',): 
                 
   for fil in tqdm(image_paths, desc=f"Prosessing images"):
      name = os.path.basename(fil)
      if name in groundtrue_files:
          masks_image = load_images(os.path.join(groundtrue_src, name))
          imagette = decouper_image(masks_image, 4)
          for j, im in enumerate(imagette[4:-4]):
              if mask == 'hsv':
                  im_mask = get_mask_from_hsv(im)
                  cv2.imwrite(os.path.join(dest_image_path, f"{data_type}/masks/{name}_{j}.png"), im_mask.astype(np.uint8))
              elif mask == 'map':
                  im_mask = get_mask_from_map(im)
                  cv2.imwrite(os.path.join(dest_image_path, f"{data_type}/masks/{name}_{j}.tiff"), im_mask.astype(np.float32))

          image = crop_images(fil,ext="tiff",
                      isdir = False, output_size=(masks_image.shape[0],masks_image.shape[1]))
          imagette = decouper_image(image, 4)
          for i, im in enumerate(imagette[4:-4]):
              cv2.imwrite(os.path.join(dest_image_path, f"{data_type}/images/{name}_{i}.png"), im)
      else:
          print(fil, groundtrue_src)

def create_dataset(dest_path: str, image_path: str,
                    grundtruth_path:str, 
                    ext: str = ".tiff",
                    test_size: float = 0.2,
                    val_size: float = 0.3):
    
    """
    Create dataset from eloncam data and groundtruth
    Args:
        dest_path (str): Path to save the dataset
        image_path (str): Path to eloncam data directory
        grundtruth_path (str): Path to eloncam groundtruth directory
        ext (str): Extension of the images
        val_size (float): Size of the validation set 0.0 - 1.0

    """
   
    image_paths, missing_gimages = get_image_paths(image_path, grundtruth_path, ext)
    
    if len(missing_gimages) > 0:
      print(f"Missing grundtruth images for: {missing_gimages}")

    trainset, data = train_test_split(image_paths, test_size=val_size, random_state=42,shuffle=True)
    valset, testset = train_test_split(data, test_size=test_size, random_state=32,shuffle=True)

    save_dataset(trainset, dest_path, 
                 os.listdir(grundtruth_path), 
                 grundtruth_path,
                 mask = 'map',
                 data_type = 'train')

    save_dataset(valset, dest_path, 
                 os.listdir(grundtruth_path), 
                 grundtruth_path,
                 mask = 'map',
                 data_type = 'val')

    save_dataset(testset, dest_path, 
                 os.listdir(grundtruth_path),
                  grundtruth_path,
                  mask = 'map',
                  data_type = 'test')
    

if __name__ == "__main__":
    create_dataset(dest_path = 'C:\\Users\\DELL\\Desktop\\Stage\\package\\dataset',
                   image_path = 'C:\\Users\\DELL\\Desktop\\Stage\\package\\data\\brute',
                   grundtruth_path = 'C:\\Users\\DELL\\Desktop\\Stage\\package\\data\\grund',
                   ext = ".tiff",
                   val_size=0.4)