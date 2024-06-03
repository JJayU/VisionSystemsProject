import cv2
from pathlib import Path
from sklearn.svm import SVC
import pickle

# This file is used to train OCR model. It reads images from 'train_letters' folder and trains the model using them.

# Path to the folder with images of characters
images_dir = Path("../train_letters")

images_paths = sorted([image_path for image_path in images_dir.iterdir() if image_path.name.endswith('.png')])

train_X = []
train_Y = []

for image_path in images_paths:
    image = cv2.imread(str(image_path), cv2.COLOR_BGR2GRAY)
    if image is None:
        print(f'Error loading image {image_path}')
        continue

    data = image.flatten()
    train_X.append(data)
    train_Y.append(image_path.name[0])

    # cv2.imshow('as', image)
    # cv2.waitKey(100)

# Train the model
model = SVC()
model.fit(train_X, train_Y)

# Save the model to file
with open('../models/model.pkl', 'wb') as f:
    pickle.dump(model, f)
