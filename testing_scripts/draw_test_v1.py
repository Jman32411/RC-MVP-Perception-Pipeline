import cv2
import numpy as np
import json
import os

# clear terminal
os.system('cls||clear')

# Load image in grayscale
data_dir = r"Data\fsoco_sample"
img_name = "amz_00075.jpg"

img_path = rf"{data_dir}\images\{img_name}"
box_path = rf"{data_dir}\bounding_boxes\{img_name}.json"

# color scalars use BGR, not RGB
red = (0, 0, 255)

with open(box_path, "r") as read_data:
    box_data = json.load(read_data)


# https://docs.opencv.org/5.0/main_modules/imgcodecs_flags.html#imread-grayscale
# Read the image into greyscale
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# convert color space out of greyscale into BGR (this allows our boxes to have color)
grey = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# Pull rectangle points
for object in box_data["objects"]:
    # TODO There is 100% a better way to do this, but I'm too lazy at the moment.
    box_bounds = object["points"]["exterior"]
    box_start = box_bounds[0]
    p1 = (int(box_start[0]), int(box_start[1]))
    box_end = box_bounds[1]
    p2 = (int(box_end[0]), int(box_end[1]))

    # Generate boxes
    cv2.rectangle(grey, p1, p2, color=red, thickness=2)

# Show image.
cv2.imshow("Greyscale", grey)
# Hold image until keypress, then delete
cv2.waitKey(0)
cv2.destroyAllWindows()