import cv2
import glob
import os
import math
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage.morphology import binary_fill_holes
from scipy import ndimage
from skimage import morphology
from skimage import img_as_ubyte

def show_image(image, gray = True):
    if gray == True:
        plt.imshow(image, cmap = "gray")
    else:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(rgb_image)
    plt.show()


def calculate_average_distance(image):
    distance_list = []
    for row in image:
        for (px, py, pz) in row:
            # Only our pixels, not added black background
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


def calculate_average_hue(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    distance_list = []
    for row in image:
        for (h, s, v) in row:
            # Only our pixels, not added black background
            if(h != 0 and s != 0 and v != 0):
                distance_list.append(h)

    # Cast list to numpy array
    distance_list = np.array(distance_list, dtype="int")

    # Calculate average
    avg = np.average(distance_list)
    return avg


def make_coin_decision(center_avg, ring_avg):
    if(center_avg < 50.0 or ring_avg < 50.0):
        decision = "Skip image"
        money = 0
    elif(center_avg < 120.0):
        if(ring_avg < 120.0):
            decision = "1 PLN"
            money = 1.00
        else:
            decision = "2 PLN"
            money = 2.00
    else:
        if(ring_avg < 120.0):
            decision = "5 PLN"
            money = 5.00
        else:
            decision = "0.50 PLN"
            money = 0.50

    return decision, money


def make_banknote_decision(avg_color):
    if(avg_color > 80.0):
        decision = "10 PLN"
        money = 10.00
    elif(avg_color > 40.0):
        decision = "100 PLN"
        money = 100.00
    else:
        decision = "50 PLN"
        money = 50.00

    return decision, money


const_colors = [ (255,0,255),       # PINK  - UNKNOWN
                 (0,255,0),         # GREEN - 0.50 PLN
                 (255,0,0),         # BLUE  - 1 PLN
                 (0,0,255),         # RED   - 2 PLN
                 (255, 255, 0),     # YELLOW - 5 PLN
                 (0, 255, 255),     # VIOLET - 10 PLN
                 (128, 55, 55),   # 50 PLN
                 (128, 255, 128),   # 100 PLN
                ]


def find_color(money):
    if(money == 0.50):
        color = const_colors[1]
    elif(money == 1.00):
        color = const_colors[2]
    elif(money == 2.00):
        color = const_colors[3]
    elif(money == 5.00):
        color = const_colors[4]
    elif(money == 10.00):
        color = const_colors[5]
    elif(money == 50.00):
        color = const_colors[6]
    elif(money == 100.00):
        color = const_colors[7]
    else:
        color = const_colors[0]
    return color[::-1]

def intersection_boolean(a,b):
  if(a[0]<=b[0]):
    w = b[0] - (a[0]+a[2])
  else:
    w = a[0] - (b[0]+b[2])
  if(a[1]<=b[1]):
    h = b[1] - (a[1]+a[3])
  else:
    h = a[1] - (b[1]+b[3])

  if w<=0 or h<=0:
    return True
  else:
    return False

def find_rectangle(img):
    max_area = (img.shape[0]-30)*(img.shape[1]-30)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    rectangle = []
    for gray in cv2.split(img):
        for thrs in range(0, 255, 26):
            if thrs == 0:
                bin = cv2.Canny(gray, 0, 50, apertureSize = 5)
                bin = cv2.dilate(bin, None)
            else:
                _retval, bin = cv2.threshold(gray, thrs, 255, cv2.THRESH_BINARY)
            bin, contours, _hierarchy = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                flag_intersection = False
                cnt_len = cv2.arcLength(cnt, True)
                cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)
                if len(cnt) == 4 and cnt.all() != 0 and cv2.contourArea(cnt) > 110000 and cv2.contourArea(cnt) < max_area and cv2.isContourConvex(cnt):
                    x, y, width, height = cv2.boundingRect(cnt)
                    r1 = (x, y, width, height);
                    for r in rectangle: # for every rectangle in results check that another rectangle intersection with it. If yes then skip it
                        xr, yr, widthr, heightr = cv2.boundingRect(r)
                        r2 = (xr, yr, widthr, heightr);
                        if intersection_boolean(r1,r2):
                            flag_intersection = True
                            break

                    if(flag_intersection == False):
                        cnt = cnt.reshape(-1, 2)
                        max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in range(4)])
                        if max_cos < 0.05:
                            rectangle.append(cnt)
    return rectangle

def find_circles(img):
    circle = []
    #out = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,33,3)
    kernel = np.ones((9,9),np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    kernel = np.ones((19,19),np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    th_1 = cv2.normalize(thresh, None, 0, 1, cv2.NORM_MINMAX)
    th_1 = binary_fill_holes(th_1)
    th_1 = morphology.remove_small_objects(th_1, 8000)
    cv_image = img_as_ubyte(th_1)
    _, contours, _ = cv2.findContours(cv_image , cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours is not None:
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, .03 * cv2.arcLength(cnt, True), True)
            vert = len(approx)
            if(3 < vert):
                area = cv2.contourArea(cnt)
                if (8000 < area <200000):
                    (cx, cy), radius = cv2.minEnclosingCircle(cnt)
                    radius=radius*0.8
                    circleArea = radius * radius * np.pi
                    ratio = area/circleArea
                    if(ratio > 0.65):
                      circle.append((int(cx),int(cy),int(radius)))
                      #cv2.circle(out, (int(cx), int(cy)), int(radius), (0,0,0), 10)
    #show_image(out,False)
    return circle

def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )


def fix_histogram(image):
    source = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    b, g, r = cv2.split(source)

    img = cv2.merge((b, g, r))
    final = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)

    return final


def check_intersection(cx, cy, cr, rx, ry, rw, rh):
    if(rx == None):
        # No intersect - no rectangle found
        return False
    else:
        # Check intersection
        circle_points = [[cx + r, cy], [cx - r, cy], [cx, cy + r], [cx, cy - r]]

        for x, y in circle_points:
            if(x >= rx and x <= (rx + rw) and y >= ry and y <= (ry + rh)):
                return True

        return False

def correctGamma(img):
  hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  imgfloat = cv2.normalize(hsv.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
  mean = np.mean(imgfloat[:,:,2])
  median = np.median(imgfloat[:,:,2])
  value = (mean+median)/2
  img2 = img/255.0
  correction = 1-(0.5-value)*1.5
  out = cv2.pow(img2,correction)
  out = np.uint8(out*255)
  return out

if __name__ == '__main__':
    results_dir = "results/"

    # Create new directory when not exists
    if not os.path.exists(results_dir):
      os.makedirs(results_dir)

    # Find files to read
    # files_name_list = glob.glob("data/picture_014*") # 1 PLN, white
    # files_name_list = glob.glob("data/picture_111*") # 10 PLN, carpet
    # files_name_list = glob.glob("data/picture_069*") # 1 PLN, table
    # files_name_list = glob.glob("data/picture_045*") # 5 PLN, carpet
    # files_name_list = glob.glob("data/picture_043*") # 0.50 PLN, white
    # files_name_list = glob.glob("data/picture_08*") + glob.glob("data/picture_09*") + glob.glob("data/picture_1*")
    # files_name_list = glob.glob("data/*") # All images
    files_name_list = glob.glob("data/bb*")
    # Read files
    image_list = list(map(cv2.imread, files_name_list))

    # Iterate on images
    for index, image in enumerate(image_list):
        all_money_list = [0]
        output = image.copy()
        overlay = image.copy()

        #show_image(image,False)
        image = fix_histogram(image)
        #show_image(image,False)
        gamma = correctGamma(image)
        #show_image(gamma,False)
        print(str(files_name_list[index]))
        # Convert image to gray
        gray = cv2.cvtColor(gamma, cv2.COLOR_BGR2GRAY)
        # Remove noise
        # removed_noise = cv2.GaussianBlur(gray, (3,3), 0)

        # Edge detection with Laplacian Derivatives
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)

        # # Calculate min, max value
        # min = math.fabs(np.amin(laplacian))
        # max = math.fabs(np.amax(laplacian))
        # sum2 = min + max #sum -> sum2 , when sum there was error with built-in function sum
        #
        # # Manually set EVERY POINT to [0; 255] - gray scale
        # gray_matrix = []
        # for line in laplacian:
        #     row = []
        #     for cell in line:
        #         value = int(((cell + min) / sum2) * 255)
        #         row.append(value)
        #
        #     gray_matrix.append(row)
        #
        # # Set matrix type, required in HoughCircles
        # gray_matrix = np.array(gray_matrix, dtype=np.uint8)
        # show_image(gray_matrix,True)
        laplacian_plus = np.fabs(laplacian)
        laplacian_normalized = cv2.normalize(laplacian_plus, None, 0, 600, cv2.NORM_MINMAX)
        laplacian_normalized[laplacian_normalized[:,:]>255]=255
        laplacian_normalized = np.array(laplacian_normalized, dtype=np.uint8)



        # Find cicles
        #circles = cv2.HoughCircles(laplacian_normalized , cv2.HOUGH_GRADIENT, 1.1, 270, param1 = 60, param2 = 100, minRadius = 95, maxRadius = 200)
        circles = find_circles(image)


        rx = None
        ry = None
        rw = None
        rh = None

        # Find banknotes
        banknote_image = image.copy()
        rectangle = find_rectangle(banknote_image)
        for img in rectangle:
            x, y, width, height = cv2.boundingRect(img)
            rx = x
            ry = y
            rw = width
            rh = height
            banknote_to_test = banknote_image[y : y + height, x : x + width].copy()

            banknote_analize = banknote_image[y + int(height / 5) : y + 4 * int(height / 5), x + int(width / 3) : x + 3 * int(width / 4)].copy()
            test_avg = calculate_average_distance(banknote_analize)
            decision, money = make_banknote_decision(test_avg)

            test_hsv = calculate_average_hue(banknote_analize)
            #print("HSV = " + str(test_hsv))
            #print("AVG = " + str(test_avg))

            cv2.rectangle(overlay, (x, y), (x + width, y + height), find_color(money), -1)
            cv2.addWeighted(overlay, 0.25, output, 0.75, 0, output)
            cv2.rectangle(output, (x, y), (x + width, y + height), find_color(money), 10)
            cv2.putText(output, "{:.2f} PLN".format(money), (np.int(x + width / 2), np.int(y + height / 2)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (204, 119, 0), 3)
            all_money_list.append(money)

        if circles is not None:
            # Convert the (x, y) coordinates and radius of the circles to integers
            # circles = np.round(circles[0, :]).astype("int")
            # Loop over the circles (x, y, r)
            for cin, (x, y, r) in enumerate(circles):
                # Crop image
                crop = image[y - r : y + r, x - r : x + r].copy()
                #show_image(crop,False)
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

                # Calculate average of distance between pixels - distance between x and y, x and z, y and z
                center_circle_avg = calculate_average_distance(center_circle)
                # print("Center = " + str(center_circle_avg))

                # Prepare ring to test outside distance
                ring = crop.copy()
                ring_radius = int(r * 0.85) # 85% of original radius
                mask = np.zeros((crop.shape[0], crop.shape[1]), dtype = np.uint8)
                cv2.circle(mask, (r, r), ring_radius, (255, 255, 255), 20, 8, 0)
                ring[mask[:,:] == 0] = 0

                ring_avg = calculate_average_distance(ring)
                # print("Ring = " + str(ring_avg))

                decision, money = make_coin_decision(center_circle_avg, ring_avg)
                # print("Decision = " + decision)
                # Draw on original image
                if(money != -1):

                    # We should also check circle not intersect rectangle (rectangle has more priority than circle)
                    intersect = check_intersection(x, y, r, rx, ry, rw, rh)

                    if not intersect:
                        cv2.circle(overlay, (x, y), r, find_color(money), -1)
                        cv2.addWeighted(overlay, 0.25, output, 0.75, 0, output)
                        cv2.circle(output, (x, y), r, find_color(money), 10)
                        cv2.putText(output, "{:.2f} PLN".format(money), (np.int(x-r/2),y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (204, 119, 0), 3)
                        all_money_list.append(money)
            cv2.putText(output, "W SUMIE: {:.2f} PLN".format(sum(all_money_list)), (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 4)
            path = results_dir + files_name_list[index].split('/')[1]
            cv2.imwrite(path, output)
