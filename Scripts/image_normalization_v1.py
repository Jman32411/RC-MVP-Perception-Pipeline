import cv2 as cv
import numpy as np
import os


def normalize_hue(in_img: np.ndarray):
    # change the data type to float32 so that numpy doesn't get angry.
    in_img = in_img.astype(np.float32)

    # separate the image into the rgb channels
    b, g, r = cv.split(in_img)

    # find the average of the 3 separate colors. This will be part of the scaling factor
    average = (b.mean() + g.mean() + r.mean())/3

    # we have to mult by the scale factor which is just the (average of the whole)/(average of the part)
    b *= average / b.mean()
    g *= average / g.mean()
    r *= average / r.mean()

    # we apply a clip function so that if (for some reason) the values DO go out of our bounds (0-255), we restrict it to the bounds.
    # Combine all values back together into their matrix
    # We also have to change the data type back to the original type
    return np.clip(cv.merge([b, g, r]), 0, 255).astype(np.uint8)


def normalize_saturation(in_img: np.ndarray):
    # convert from BGR to HSV
    # change the data type to float32 so that numpy doesn't get angry.
    in_img = cv.cvtColor(in_img, cv.COLOR_BGR2HSV).astype(np.float32)

    # separate the image into the rgb channels
    h, s, v = cv.split(in_img)

    # find the average of the 3 separate colors. This will be part of the scaling factor

    # we have to mult by the scale factor which is just the (average of the whole)/(average of the part)
    print(s.mean())
    print(v.mean())

    s *= s/s.mean()
    v *= s/v.mean()
    
    print(s)
    print(v)

    s = np.clip(s, 0, 255)
    v = np.clip(v, 0, 255)
    print(s)
    print(v)
    # we apply a clip function so that if (for some reason) the values DO go out of our bounds (0-179 for H; 0-255 for S and V), we restrict it to the bounds.
    # Combine all values back together into their matrix
    # We also have to change the data type back to the original type
    return cv.cvtColor(cv.merge([h, s, v]).astype(np.uint8), cv.COLOR_HSV2BGR)


# clear terminal
os.system('cls||clear')

data_dir = r"Data\fsoco_sample"
img_name = "amz_00075.jpg"
img_name = "amz_00016.jpg"
img_name = "amz_00588.jpg"

img_path = rf"{data_dir}\images\{img_name}"

img = cv.imread(img_path, cv.IMREAD_COLOR_BGR)

norm_img = normalize_hue(img)
norm_img = normalize_saturation(norm_img)
img_scale = cv.resize(norm_img, (940, 540))
# convert to HSV as that's better for edge detection
hsv = cv.cvtColor(img_scale, cv.COLOR_BGR2HSV)

#* REMINDER OpenCV uses H: 0-179, S: 0-255, V: 0-255
lower_bound = np.array([22, 95, 167])
upper_bound = np.array([50, 255, 255])
filter = cv.inRange(hsv, lower_bound, upper_bound)

# use a bitwise and to look for points in our image that ONLY fall between our bounds
masked = cv.bitwise_and(img_scale, img_scale, mask=filter)

# Show image.
cv.imshow("original", img_scale)
# cv.imshow("B/W filter", filter)
cv.imshow("Overlaid filter", masked)
# Hold image until keypress, then delete
cv.waitKey(0)
cv.destroyAllWindows()