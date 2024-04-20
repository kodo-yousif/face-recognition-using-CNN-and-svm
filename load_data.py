import os
import cv2
import numpy as np
from skimage.filters import sobel

def load_data(directory):
    X = []
    y = []
    label = 0
    
    # Iterate through each person's folder
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".bmp"):
                filepath = os.path.join(subdir, file)
                image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

                image = image - sobel(image)
                
                X.append(image)
                y.append(label)
        label += 1
    
    X = np.array(X, dtype='float32')
    y = np.array(y)
    
    # Normalize the image data to 0-1
    X /= 255.0
    
    X_expanded = np.expand_dims(X, axis=-1)

    # X_resized = np.array([cv2.resize(img, (64, 64)) for img in X]).reshape(-1, 64, 64, 1)

    return X_expanded, y