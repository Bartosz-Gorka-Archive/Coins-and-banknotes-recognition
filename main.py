import cv2
import glob
import os
import numpy as np
from matplotlib import pyplot as plt


def show_image(image, gray = True):
    b, g, r = cv2.split(image)
    rgb_image = cv2.merge([r,g,b])

    if gray == True:
        plt.imshow(rgb_image, cmap = plt.cm.gray)
    else:
        plt.imshow(rgb_image)

    plt.show()


def detect_edges(image):
    return image


if __name__ == '__main__':
    results_dir = "results/"

    # Create new directory when not exists
    if not os.path.exists(results_dir):
      os.makedirs(results_dir)

    # Find files to read
    files_name_list = glob.glob("data/picture_003*")
    # files_name_list = glob.glob("data/*")

    # Read files
    image_list = list(map(cv2.imread, files_name_list))

    # Iterate on images
    for image in image_list:
        show_image(image)
