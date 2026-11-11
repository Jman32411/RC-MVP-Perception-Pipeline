import cv2
import numpy as np
 
# Load image in grayscale
# https://docs.opencv.org/5.0/main_modules/imgcodecs_flags.html#imread-grayscale
img = cv2.imread(r"Data\fsoco_sample\images\amz_00016.jpg", cv2.IMREAD_GRAYSCALE)

cv2.imshow("Sobel Edge Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()