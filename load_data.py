import os
import cv2
import numpy as np

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
                X.append(image)
                y.append(label)
        label += 1
    
    X = np.array(X, dtype='float32')
    y = np.array(y)
    
    # Normalize the image data to 0-1
    X /= 255.0
    
    return X, y