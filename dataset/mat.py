
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgb2hsv, rgb2gray
from skimage.morphology import skeletonize
import os
from scipy.ndimage import distance_transform_edt

def load_images(path, isdir = True, ext = '.tiff') -> list[np.ndarray]:
    """Load images from a directory or a single image file.

    Args:
        path (str): Path to the directory or image file.
        isdir (bool): Whether the path is a directory. Default is True.
        ext (str): Extension of image files to load if path is a directory. Default is '.tif'.

    Returns:
        list: List of loaded images as numpy arrays.
    """
    images = []
    if isdir:
        for fname in os.listdir(path):
            if fname.endswith(ext):
                img = imread(os.path.join(path, fname))
                images.append(img)
    else:
        img = imread(path)
        images.append(img)
    return images



def extract_skeleton(rgbimage: np.ndarray, axis = 1) -> np.ndarray:
    """Extract skeleton from a RGB image.

    Args:
        rgbimage (np.ndarray): Input RGB image.
    
    Returns:
        np.ndarray: Skeletonized image.
    """
    hsv_image = rgb2hsv(rgbimage)
    saturation = hsv_image[:, :, axis]
    skeleton = skeletonize(saturation.astype(np.bool_))
    return skeleton



def distance_map(img,) ->np.ndarray:
    """Compute the distance map of the binary image.

    Args:
        image (np.ndarray): Input binary image.
        save_path (str): Path to save the distance map. If None, the map is returned. Default is None.
        ext (str): Extension of image files to load if path is a directory. Default is '.tif'.

    Returns:
        list: List of distance maps or paths to saved distance maps.
    """
    skeleton = extract_skeleton(img)

    skeleton = skeleton.astype(np.bool_)

    # Distance euclidienne au squelette (0 sur le squelette)
    D = distance_transform_edt(~skeleton)

    # Normalisation (utile pour U-Net)
    D = D / (D.max() + 1e-8)

    return D

if __name__ == "__main__":
    image_pth = "Aker_10____1710201848_5B1_057.tiff"

    img = imread(image_pth)
    D = distance_map(img=img)

    #plt.plot(D[100, :])
    plt.imshow(D, cmap='gray')
    plt.show()

    