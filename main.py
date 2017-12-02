import cv2
import glob


if __name__ == '__main__':
    results_dir = "results/"

    # Create new directory when not exists
    if not os.path.exists(results_dir):
      os.makedirs(results_dir)

    # Find files to read
    files_name_list = glob.glob("data/*")

    # Read files
    image_list = list(map(cv2.imread, files_name_list))
