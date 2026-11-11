import cv2 as cv
import numpy as np
import os

def nothing(x):
   pass

# clear terminal
os.system('cls||clear')

data_dir = r"Data\fsoco_sample"
img_name = "amz_00075.jpg"
# img_name = "ff_00258.png"

img_path = rf"{data_dir}\images\{img_name}"

img = cv.imread(img_path, cv.IMREAD_ANYCOLOR)
img_scale = cv.resize(img, (940, 540))
hsv = cv.cvtColor(img_scale, cv.COLOR_BGR2HSV)

# Generate a window "container"
window_name = "Overlaid filter"
cv.namedWindow(window_name)

# Put the sliders into the window
#* REMINDER OpenCV uses H: 0-179, S: 0-255, V: 0-255
cv.createTrackbar('H_lower', window_name, 0, 179, nothing)
cv.createTrackbar('S_lower', window_name, 0, 255, nothing)
cv.createTrackbar('V_lower', window_name, 0, 255, nothing)

cv.createTrackbar('H_upper', window_name, 0, 179, nothing)
cv.createTrackbar('S_upper', window_name, 0, 255, nothing)
cv.createTrackbar('V_upper', window_name, 0, 255, nothing)


key_pressed = cv.waitKey(1)

while key_pressed == -1:
    # Get the slider positions and apply them to the bounds
    lower_bound = np.array([int(cv.getTrackbarPos('H_lower',window_name)), int(cv.getTrackbarPos('S_lower',window_name)), int(cv.getTrackbarPos('V_lower',window_name))])
    upper_bound = np.array([int(cv.getTrackbarPos('H_upper',window_name)), int(cv.getTrackbarPos('S_upper',window_name)), int(cv.getTrackbarPos('V_upper',window_name))])

    # Apply the filter
    filter = cv.inRange(hsv, lower_bound, upper_bound)

    # use a bitwise and to look for points in our image that ONLY fall between our bounds
    masked = cv.bitwise_and(img_scale, img_scale, mask=filter)

    # Show image.
    cv.imshow("original", img_scale)

    # show filtered image
    cv.imshow("Overlaid filter", masked)

    # Check periodically to see if any key is pressed.
    key_pressed = cv.waitKey(1)