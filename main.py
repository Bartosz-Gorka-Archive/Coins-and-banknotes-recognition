import cv2
import glob
import os
import math
import numpy as np
from matplotlib import pyplot as plt


def show_image(image, gray = True):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if gray == True:
        plt.imshow(rgb_image, cmap = "gray")
    else:
        plt.imshow(rgb_image)

    plt.show()


def calculate_average_distance(image):
    distance_list = []
    for row in image:
        for (px, py, pz) in row:
            # Onlyour pixels, not added black background
            if(px != 0 and py != 0 and pz != 0):
                # Calculate distance
                value = abs(int(px) - int(py)) + abs(int(px) - int(pz)) + abs(int(pz) - int(py))

                # Append calculated value
                distance_list.append(value)

    # Cast list to numpy array
    distance_list = np.array(distance_list, dtype="int")

    # Calculate average
    avg = np.average(distance_list)
    return avg


def make_decision(center_avg, ring_avg):
    if(center_avg < 50.0 or ring_avg < 50.0):
        decision = "Skip image"
    elif(center_avg < 110.0):
        if(ring_avg < 110.0):
            decision = "1 PLN"
        else:
            decision = "2 PLN"
    else:
        if(ring_avg < 110.0):
            decision = "5 PLN"
        else:
            decision = "0.50 PLN"

    return decision


if __name__ == '__main__':
    results_dir = "results/"

    # Create new directory when not exists
    if not os.path.exists(results_dir):
      os.makedirs(results_dir)

    # Find files to read
    # files_name_list = glob.glob("data/picture_014*") # 1.00
    files_name_list = glob.glob("data/picture_045*") # 5.00, carpet
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
        circles = cv2.HoughCircles(gray_matrix, cv2.HOUGH_GRADIENT, 1.2, 150, param1 = 100, param2 = 100, minRadius = 90, maxRadius = 200)

        if circles is not None:
            # Convert the (x, y) coordinates and radius of the circles to integers
            circles = np.round(circles[0, :]).astype("int")

            # Loop over the circles (x, y, r)
            for cin, (x, y, r) in enumerate(circles):
                # Crop image
                crop = image[y - r : y + r, x - r : x + r].copy()

                # Set mask, black background
                mask = np.zeros((crop.shape[0], crop.shape[1]), dtype = np.uint8)
                cv2.circle(mask, (r, r), r, (255, 255, 255), -1, 8, 0)
                crop[mask[:,:] == 0] = 0

                # Prepare center circle radius
                center_radius = int(r * 0.1) # 10% of original radius

                # Copy original image to new variable
                center_circle = image[y - center_radius : y + center_radius, x - center_radius : x + center_radius].copy()

                # Prepare mask with zeros to cut only circle
                mask = np.zeros((center_circle.shape[0], center_circle.shape[1]), dtype = np.uint8)

                # Cut circle from center
                cv2.circle(mask, (center_radius, center_radius), center_radius, (255, 255, 255), -1, 8, 0)

                # Add black background outside
                center_circle[mask[:,:] == 0] = 0

                # Show cut center circle
                # show_image(center_circle)

                # Calculate average of distance between pixels - distance between x and y, x and z, y and z
                center_circle_avg = calculate_average_distance(center_circle)
                print("Center = " + str(center_circle_avg))

                # Prepare ring to test outside distance
                ring = crop.copy()
                mask = np.zeros((crop.shape[0], crop.shape[1]), dtype = np.uint8)
                cv2.circle(mask, (r, r), r, (255, 255, 255), 20, 8, 0)
                ring[mask[:,:] == 0] = 0

                ring_avg = calculate_average_distance(ring)
                print("Ring = " + str(ring_avg))

                decision = make_decision(center_circle_avg, ring_avg)
                print("Decision = " + decision)

                path = results_dir + str(cin) + files_name_list[index].split('/')[1]
                cv2.imwrite(path, crop)
