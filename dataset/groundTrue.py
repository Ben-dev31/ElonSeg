

import numpy as np
import cv2

def create_ground_truth_mask(image, saved = bool, channel=1, **kwargs):
    img = None 

    if isinstance(image, str):
        im = cv2.imread(image)
        img = np.array(im)
    elif isinstance(image, np.ndarray):
        img = image.copy()
    
    if img is not None:
       hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) 
       mask = hsv[:, :, channel] 
    else:
        raise ValueError("Invalid image input. Provide a file path or a numpy array.")
    
    if saved:
        file_name = kwargs.get("file_name", "mask.png")
        cv2.imwrite(file_name, mask)
    else:
        return mask

if __name__ == "__main__":
    # Example usage
    import matplotlib.pyplot as plt
    mask = create_ground_truth_mask("./../Aker_10____1710200648_5B2_054.tif", saved=False, channel=1)
    plt.imshow(mask, cmap='gray')
    plt.show()