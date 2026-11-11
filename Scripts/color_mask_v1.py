import cv2 as cv
import numpy as np
import os

# clear terminal
os.system('cls||clear')

data_dir = r"Data\fsoco_sample"
img_name = "amz_00075.jpg"
# img_name = "ff_00258.png"

img_path = rf"{data_dir}\images\{img_name}"

img = cv.imread(img_path, cv.IMREAD_ANYCOLOR)
img_scale = cv.resize(img, (940, 540))
# convert to HSV as that's better for edge detection
hsv = cv.cvtColor(img_scale, cv.COLOR_BGR2HSV)

#* REMINDER OpenCV uses H: 0-179, S: 0-255, V: 0-255
lower_bound = np.array([22, 95, 167])
upper_bound = np.array([50, 255, 255])
while True:
    filter = cv.inRange(hsv, lower_bound, upper_bound)

    # use a bitwise and to look for points in our image that ONLY fall between our bounds
    masked = cv.bitwise_and(img_scale, img_scale, mask=filter)

    # Show image.
    cv.imshow("original", img_scale)
    # cv.imshow("B/W filter", filter)
    cv.imshow("Overlaid filter", masked, )
    # Hold image until keypress, then delete
    cv.waitKey(0)
    cv.destroyWindow("Overlaid filter")

    
    os.system('cls||clear')
    print(f"previous lower was {lower_bound}")
    print(f"previous upper was {upper_bound}")
    lower_bound = []
    upper_bound = []
    for i in range(0, 3):
        inval = int(input(f"lower bound {i}: "))
        lower_bound.append(inval)
    lower_bound = np.array(lower_bound)
    
    for i in range(0, 3):
        inval = int(input(f"Upper bound {i}: "))
        upper_bound.append(inval)
    upper_bound = np.array(upper_bound)
    
    cv.destroyWindow("original")
