
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import medial_axis
from skimage.io import imread, imsave
from skimage.filters import threshold_otsu, threshold_local
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage.morphology import skeletonize, opening, closing, dilation, erosion
from skimage.color import rgb2hsv, rgb2gray
from skimage.filters import median
from tqdm import tqdm
import networkx as nx
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

def binarize_image(image: np.ndarray, method: str = 'otsu', block_size: int = 35) -> np.ndarray:
    """Binarize an image using specified thresholding method.

    Args:
        image (np.ndarray): Input image.
        method (str): Thresholding method ('otsu' or 'local'). Default is 'otsu'.
        block_size (int): Block size for local thresholding. Default is 35.

    Returns:
        np.ndarray: Binarized image.
    """
    gray_image = rgb2gray(image)
    if method == 'otsu':
        thresh = threshold_otsu(gray_image)
        binary = gray_image < thresh
    elif method == 'local':
        thresh = threshold_local(gray_image, block_size)
        binary = gray_image < thresh
    else:
        raise ValueError("Unsupported method. Use 'otsu' or 'local'.")
    return binary.astype(np.uint8) * 255

def extract_skeleton(rgbimage: np.ndarray, axis = 1) -> np.ndarray:
    """Extract skeleton from a RGB image.

    Args:
        rgbimage (np.ndarray): Input RGB image.
    
    Returns:
        np.ndarray: Skeletonized image.
    """
    hsv_image = rgb2hsv(rgbimage)
    saturation = hsv_image[:, :, axis]
    skeleton = skeletonize(saturation.astype(np.bool))
    return skeleton

def skeleton_to_graph(skeleton) -> nx.Graph:
    """Convert a skeletonized image to a graph representation."""

    G = nx.Graph()
    ys, xs = np.where(skeleton)

    for y, x in zip(ys, xs):
        G.add_node((y, x))
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx_ = y + dy, x + dx
                if (ny, nx_) in G:
                    G.add_edge((y, x), (ny, nx_))
    return G

def transform_to_nodes(binary: np.ndarray, G: nx.Graph) -> list[tuple[int, int]]:
    """Transform skeleton pixels to a list of node coordinates.

    Args:
        binary (np.ndarray): Binary image.
        G (nx.Graph): Graph representation of the skeleton.

    Returns:
        list: List of (y, x) coordinates of skeleton pixels.
    """
    dist_edge = distance_transform_edt(binary)
    r_raw = {node: dist_edge[node] for node in G.nodes}

    return r_raw

def smooth_radius_on_graph(G, r_raw, n_iter=20, alpha=0.6):
    """Smooth radius values on the graph using neighbor averaging.
    
    Args:
        G (nx.Graph): Graph representation of the skeleton.
        r_raw (dict): Raw radius values for each node.
        n_iter (int): Number of iterations for smoothing. Default is 20.
        alpha (float): Weight for the current value in smoothing. Default is 0.6.

    Returns:
        dict: Smoothed radius values for each node.
    """
    r = r_raw.copy()
    for _ in range(n_iter):
        r_new = {}
        for node in G.nodes:
            neigh = list(G.neighbors(node))
            if len(neigh) == 0:
                r_new[node] = r[node]
            else:
                mean_neigh = np.mean([r[n] for n in neigh])
                r_new[node] = alpha * r[node] + (1 - alpha) * mean_neigh
        r = r_new
    return r

def create_radius_map(path: str, isdir: bool = True, ext=".tiff", sigma = 2.5, save_path: str = None) -> np.ndarray:
    """Create a radius map from smoothed radius values.

    Args:
        binary (np.ndarray): Binary image.
        r_smooth (dict): Smoothed radius values for each node.

    Returns:
        np.ndarray: Radius map.
    """
    images = load_images(path, isdir=isdir, ext=ext)
    
    mats = []

    for image in tqdm(images, desc="Creating radius maps"):
        binary = binarize_image(image)
        skeleton = extract_skeleton(image)
        r_raw = transform_to_nodes(binary, skeleton_to_graph(skeleton))

        G = skeleton_to_graph(skeleton)
        r_smooth = smooth_radius_on_graph(G, r_raw)

        dist_to_skel = distance_transform_edt(~skeleton)

        # carte sparse du rayon
        radius_sparse = np.zeros_like(binary, dtype=float)
        for (y, x), r in r_smooth.items():
            radius_sparse[y, x] = r

        # largeur de diffusion radiale

        radius_map = np.zeros_like(radius_sparse)

        mask = dist_to_skel <= radius_sparse.max()

        radius_map[mask] = (
            np.exp(-dist_to_skel[mask] / sigma)
            * radius_sparse.max()
        )

        # contraindre à l'objet
        radius_map *= binary

        if save_path is not None:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            base_name = os.path.basename(path) if not isdir else os.path.basename(image)
            save_fname = os.path.join(save_path, base_name)
            imsave(save_fname, radius_map.astype(np.float32))
        else:
            mats.append(radius_map)

    return mats

def distance_map(path: str, 
                 isdir: bool = True, 
                 ext=".tiff", 
                 save_path: str = None) -> list[np.ndarray]:
    """Compute the distance map of the binary image.

    Args:
        image (np.ndarray): Input binary image.
        save_path (str): Path to save the distance map. If None, the map is returned. Default is None.
        ext (str): Extension of image files to load if path is a directory. Default is '.tif'.

    Returns:
        list: List of distance maps or paths to saved distance maps.
    """
    images = load_images(path, isdir=isdir, ext=ext)
    
    dist_maps = []

    for idx, image in enumerate(tqdm(images, desc="Computing distance maps")):
        skeleton = extract_skeleton(image)

        skeleton = skeleton.astype(bool)

        # Distance euclidienne au squelette (0 sur le squelette)
        D = distance_transform_edt(~skeleton)

        # Normalisation (utile pour U-Net)
        D = D / (D.max() + 1e-8)

        if save_path is not None:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            base_name = os.path.basename(path) if not isdir else os.path.basename(image)
            save_fname = os.path.join(save_path, base_name)
            imsave(save_fname, D.astype(np.float32))
            dist_maps.append(save_fname)
        else:
            dist_maps.append(D)

    return dist_maps

if __name__ == "__main__":
    image_pth = "C:\\Users\\DELL\\Desktop\\Stage\\DATA\\Prepared_data\\Analysed_data\\Aker_10____1710191850_5B4_051.tiff"

    D = distance_map(image_pth, isdir=False)[0]

    #plt.plot(D[100, :])
    plt.imshow(D, cmap='gray')
    plt.show()

    