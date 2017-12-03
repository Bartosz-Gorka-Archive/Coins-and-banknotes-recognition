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
    files_name_list = glob.glob("data/picture_014*") # 1.00
    # files_name_list = glob.glob("data/picture_043*") # 0.50
    # files_name_list = glob.glob("data/*") # all

    # Read files
    image_list = list(map(cv2.imread, files_name_list))

    # Iterate on images
    for index, image in enumerate(image_list):
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
            for cin, (x, y, r) in enumerate(circles):
                # Crop image
                crop = image[y-r:y+r,x-r:x+r].copy()

                # Set mask, black background
                mask = np.zeros((crop.shape[0],crop.shape[1]),dtype=np.uint8)
                cv2.circle(mask,(r,r),r, (255,255,255), -1, 8, 0)
                crop[mask[:,:]==0]=0

                # Resize cropped image
                # resized_image = cv2.resize(crop, (128, 128))

                # Show image
                new_r = int(r*0.1)
                new = image[y-new_r:y+new_r,x-new_r:x+new_r].copy()
                mask = np.zeros((new.shape[0],new.shape[1]),dtype=np.uint8)
                cv2.circle(mask,(new_r,new_r),new_r, (255,255,255), -1, 8, 0)
                new[mask[:,:]==0]=0
                show_image(new)

                different_colors_list = []
                for row in new:
                    for (px, py, pz) in row:
                        if(px != 0 and py != 0 and pz != 0):
                            value = abs(int(px) - int(py)) + abs(int(px) - int(pz)) + abs(int(pz) - int(py))
                            different_colors_list.append(value)

                different_colors_list = np.array(different_colors_list, dtype="int")
                print("Average = " + str(np.average(different_colors_list)))

                path = results_dir + str(cin) + files_name_list[index].split('/')[1]
                cv2.imwrite(path, crop)
