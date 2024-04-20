import os
import cv2
import numpy as np
from collections import Counter
from imblearn.over_sampling import SMOTE

directory = 'ORLDatabase-2021'
new_directory = "over_sampled"

if not os.path.exists(new_directory):
    os.makedirs(new_directory)
    
images = []
labels = []
label = 0
strategy = {}
target_oversampling = 25

for subdir, dirs, files in os.walk(directory):
    for file in files:
        if file.endswith(".bmp"):
            filepath = os.path.join(subdir, file)
            image = cv2.imread(filepath, cv2.IMREAD_COLOR)
            images.append(image.reshape(-1))
            labels.append(label)
            strategy[label] = target_oversampling
    label += 1


if len(labels) != 0:
    images = np.array(images)
    labels = np.array(labels)

    smote = SMOTE(sampling_strategy=strategy, random_state=42)
    images_resampled, labels_resampled = smote.fit_resample(images, labels)

    
    for i, image in enumerate(images_resampled):
        image_reshaped = image.reshape((112, 92, 3))

        folder_path = new_directory+ f"/{labels_resampled[i]}"

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        output_path = os.path.join(folder_path, f"oversampled_{i+1}.bmp")
        
        cv2.imwrite(output_path, image_reshaped)



