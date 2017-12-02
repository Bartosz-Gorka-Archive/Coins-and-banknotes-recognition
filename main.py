import cv2
import glob
import os
import math
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
    #     # show_image(image)

        # Convert image to gray
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Remove noise
        # removed_noise = cv2.GaussianBlur(gray, (3,3), 0)

        # Edge detection with Laplacian Derivatives
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)

        # Calculate min, max value
        min = math.fabs(np.amin(laplacian))
        max = math.fabs(np.amax(laplacian))
        sum = min + max

        # Manually set EVERY POINT to [0; 255] - gray scale
        gray_matrix = []
        for line in laplacian:
            row = []
            for cell in line:
                value = int(((cell + min) / sum) * 255)
                row.append(value)

            gray_matrix.append(row)

        # Set matrix type, required in HoughCircles
        gray_matrix = np.array(gray_matrix, dtype=np.uint8)

        # Find cicles
        circles = cv2.HoughCircles(gray_matrix, cv2.HOUGH_GRADIENT, 1.2, 150, param1=100, param2=100, minRadius=90, maxRadius=200)

        if circles is not None:
            # Convert the (x, y) coordinates and radius of the circles to integers
            circles = np.round(circles[0, :]).astype("int")

            # Loop over the circles (x, y, r)
            for (x, y, r) in circles:
                # Crop image
                crop = image[y-r:y+r,x-r:x+r].copy()

                # Set mask, black background
                mask = np.zeros((crop.shape[0],crop.shape[1]),dtype=np.uint8)
                cv2.circle(mask,(r,r),r, (255,255,255), -1, 8, 0)
                crop[mask[:,:]==0]=0

                # Resize cropped image
                resized_image = cv2.resize(crop, (128, 128))

                # Show image
                show_image(resized_image)
