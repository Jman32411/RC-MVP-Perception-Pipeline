import cv2
import numpy as np
import json
import os

# clear terminal
os.system('cls||clear')

# Load image in grayscale
data_dir = r"Data\fsoco_sample"
img_name = "amz_00016.jpg"

img_path = rf"{data_dir}\images\{img_name}"
box_path = rf"{data_dir}\bounding_boxes\{img_name}.json"

red = (255, 0, 0)

with open(box_path, "r") as read_data:
    box_data = json.load(read_data)


# https://docs.opencv.org/5.0/main_modules/imgcodecs_flags.html#imread-grayscale
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
for object in box_data["objects"]:
    box_bounds = object["points"]["exterior"]
    box_start = box_bounds[0]
    p1 = (int(box_start[0]), int(box_start[1]))
    box_end = box_bounds[1]
    p2 = (int(box_end[0]), int(box_end[1]))
    cv2.rectangle(img, p1, p2, red, 2)


cv2.imshow("Greyscale", img)
cv2.waitKey(0)
cv2.destroyAllWindows()