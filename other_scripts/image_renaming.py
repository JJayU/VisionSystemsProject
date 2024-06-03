import cv2
from pathlib import Path

# Script for renaming images from selected folder
# It shows the image and asks for the license plate number and saves the image with the name of the license plate
# Useful for preparing test dataset

# Folder containing images to rename
images_dir = Path("../train_2")

# Folder to save renamed images
output_dir = "../train_3/"

images_paths = sorted([image_path for image_path in images_dir.iterdir()])

# Iterate over images and ask the user for the license plate number, then save it with the name of the license plate
for image_path in images_paths:
    img = cv2.imread(str(image_path))

    cv2.namedWindow('Tablica', cv2.WINDOW_NORMAL)
    cv2.imshow('Tablica', img)
    cv2.resizeWindow('Tablica', 800, 600)

    cv2.waitKey(1)

    name = input('Podaj tablice: ')
    filename = output_dir + str(name) + '.jpg'

    cv2.imwrite(filename, img)
