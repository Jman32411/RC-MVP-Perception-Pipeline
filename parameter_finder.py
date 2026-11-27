import cv2 as cv
import numpy as np
import os
import json
import sys

def nothing(x):
   pass


def normalize_hue(in_img: np.ndarray, scale_factor: float):
    # change the data type to float32 so that numpy doesn't get angry.
    in_img = in_img.astype(np.float32)

    # separate the image into the rgb channels
    b, g, r = cv.split(in_img)

    # find the average of the 3 separate colors. This will be part of the scaling factor
    average = (b.mean() + g.mean() + r.mean())/3

    # we have to mult by the scale factor which is just the (average of the whole)/(average of the part)
    b *= scale_factor * average / b.mean()
    g *= scale_factor * average / g.mean()
    r *= scale_factor * average / r.mean()

    # we apply a clip function so that if (for some reason) the values DO go out of our bounds (0-255), we restrict it to the bounds.
    # Combine all values back together into their matrix
    # We also have to change the data type back to the original type
    return np.clip(cv.merge([b, g, r]), 0, 255).astype(np.uint8)

def normalize_saturation(in_img: np.ndarray, s_scale: float, v_scale):
    # convert from BGR to HSV
    # change the data type to float32 so that numpy doesn't get angry.
    in_img = cv.cvtColor(in_img, cv.COLOR_BGR2HSV).astype(np.float32)

    # separate the image into the hsv channels
    h, s, v = cv.split(in_img)

    # scale by the average for both S and V
    s *= s_scale*s/s.mean()
    v *= v_scale*v/v.mean()

    s = np.clip(s, 0, 255)
    v = np.clip(v, 0, 255)
    # we apply a clip function so that if (for some reason) the values DO go out of our bounds (0-179 for H; 0-255 for S and V), we restrict it to the bounds.
    # Combine all values back together into their matrix
    # We also have to change the data type back to the original type
    return cv.cvtColor(cv.merge([h, s, v]).astype(np.uint8), cv.COLOR_HSV2BGR)

def get_mask(image_mat: np.ndarray, hue_scale: float, sat_scale: float, val_scale: float, lower_bound: list, upper_bound: list):
    # apply the normalization to HSV
    norm_img = normalize_hue(image_mat, hue_scale)
    norm_img = normalize_saturation(norm_img, sat_scale, val_scale)

    hsv = cv.cvtColor(norm_img, cv.COLOR_BGR2HSV)

    lower_bound = np.array(lower_bound)
    upper_bound = np.array(upper_bound)

    # get the filter
    filter = cv.inRange(hsv, lower_bound, upper_bound)

    # use a bitwise and to look for points in our image that ONLY fall between our bounds
    mask = cv.bitwise_and(norm_img, norm_img, mask=filter)
    return mask

def find_contours(image_mat: np.ndarray):
    # Apply a grey coloring to the mask (to make it all one color)
    grey = cv.cvtColor(image_mat, cv.COLOR_BGR2GRAY)

    # Threshold based on brightness values (the range being the upper half of 255)
    # Arbitrarilly chose Triangle over OTSU as there shouldn't be much of an impact using one over the other 
    ret, thresh = cv.threshold(grey, 127, 255, cv.THRESH_TRIANGLE)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    return contours

# courtesy of claude
def merge_cone_boxes(boxes, max_gap=0.5, max_overlap=0.2, tol=2):
    """
    boxes: list of (x, y, w, h)
    A small box is paired with a larger box below it only if the small box's
    left AND right edges both lie within the larger box's left/right edges
    (give or take `tol` pixels).
    """
    b = np.asarray(boxes, dtype=np.float32)
    if len(b) == 0:
        return []

    x, y, w, h = b.T
    x2, y2 = x + w, y + h
    area = w * h
    used = np.zeros(len(b), bool)
    merged = []

    for i in np.argsort(area):          # smallest (top) boxes first
        if used[i]:
            continue

        gap = y - y2[i]                 # top box bottom -> candidate top

        cand = (~used) & (area > area[i]) & (y > y[i])

        # horizontal containment: top box fully inside candidate's x-range
        cand &= (x[i] >= x - tol) & (x2[i] <= x2 + tol)

        # vertical relationship: small gap, or slight overlap
        cand &= (gap < max_gap * h) & (gap > -max_overlap * h)

        if not cand.any():
            continue

        idx = np.flatnonzero(cand)
        j = idx[np.argmin(np.abs(gap[idx]))]   # nearest vertically

        nx1, ny1 = min(x[i], x[j]), min(y[i], y[j])
        nx2, ny2 = max(x2[i], x2[j]), max(y2[i], y2[j])
        merged.append((int(nx1), int(ny1), int(nx2 - nx1), int(ny2 - ny1)))
        used[i] = used[j] = True

    merged += [tuple(map(int, b[k])) for k in np.flatnonzero(~used)]
    return merged

def get_cone_boxes(contours: np.ndarray, horizon: int, error: float, cone_scale_factor: float, color: str, size: str):
    if size == "small":
        # initialize empty lists
        rect_bounds = []
        min_height_scalar = 0.01
        min_width_scalar = 0.003
        min_height = round(frame_height*min_height_scalar)
        min_width = round(frame_width*min_width_scalar)
        min_area_ratio = 0.27
        max_aspect_ratio = 2
        min_aspect_ratio = 0.2
        # print(f"Frame: {img_width}x{img_height}")
        # print(f"min_width: {min_width} | min_height: {min_height} | horizon_Y: {horizon}")

        for contour in contours:
            # generate the bounding rectangles around each polygonal contour
            # Returns (top left corner) x, y, width, height
            contour_area = cv.contourArea(contour)
            rectangle_info = cv.boundingRect(contour)

            # calculate the area using the width and height values
            width = rectangle_info[2]
            height = rectangle_info[3]
            area = height*width
            rect_aspect_ratio = width/height

            # Filter the rectangles that are above our horizon line
            if (rectangle_info[1]+height<=horizon):
                continue
            
            # filter the rectangles that are too thin or flat (size)
            if (height < min_height) or (width < min_width):
                continue

            # Filter the rectangles that are too wide/thin (ratio)
            if (rect_aspect_ratio>= max_aspect_ratio) or (rect_aspect_ratio <= min_aspect_ratio):
                continue

            # Filter based on contour/rectangle area (actual color ratio of segment)
            # if (contour_area/area < min_area_ratio):
            #     continue

            # If they fit the criteria add it to the list
            rect_bounds.append(rectangle_info)

            # print(f"W: {width} | H: {height} |AR: {rect_aspect_ratio}")
            # print(f"rect_Area:{area} | cont_area: {contour_area} | cont/rect: {contour_area/area}")
            # print(f"Y: {rectangle_info[1]} | %: {((rectangle_info[1]-horizon)/(img_height-(2*y_border)))*100}")

        # Merge above boxes with the ones below
        rect_bounds = merge_cone_boxes(rect_bounds, 1, 1, 10)

        # As much as I hate it, to effectively utilize the scaling from the horizon, we need to do a second round of filtering
        filtered_rect_bounds = []
        for bound in rect_bounds:
            # check to see if the the bounding box height is larger or smaller than (scalefactor * height+y_border) +- error  
            # Stricter for for the lower bound
            # if (((cone_scale_factor+error)*(bound[1]-y_border))<bound[3]) or (((cone_scale_factor-(error/2.5))*(bound[1]-y_border))>bound[3]):
            #     continue


        # bounding box follows the format: {"color": ____, "box": [x, y, width, height]}

            filtered_rect_bounds.append(
                {
                    "color": color,
                    "box": bound,
                }
            )
    return(filtered_rect_bounds)

def draw_boxes(input_image: np.ndarray, box_data: list, index: int):
    thickness = 2
    color_key = {
        "blue":(255, 0, 0),
        "yellow": (0, 255, 255),
    }
    # Each frame/box_data should be a list of dictionaries
    # each dictionary is ONE bounding box that follows this format {"color": ____, "box": [x, y, width, height]}
    cv.setTrackbarMax('box_index', file_name, len(box_data))
    if (index-1) < 0:
        sys.stdout.write("\033c")
        print("Please use the \"box_index\" slider to select a full-sized cone to use as your scale-factor measurement")
        print("You can press any key on your keyboard to advance to the next frame/image (if no slider is selected).")
        for box in box_data:
            color = box["color"]
            bounds = box["box"]
            
            # draw the rect
            # top left corner
            cv.line(input_image, (bounds[0], bounds[1]), (bounds[0]+bounds[2], bounds[1]), color_key[color], thickness)
            cv.line(input_image, (bounds[0], bounds[1]), (bounds[0], bounds[1]+bounds[3]), color_key[color], thickness)

            # bottom right corner
            cv.line(input_image, (bounds[0]+bounds[2], bounds[1]+bounds[3]), (bounds[0]+bounds[2], bounds[1]), color_key[color], thickness)
            cv.line(input_image, (bounds[0]+bounds[2], bounds[1]+bounds[3]), (bounds[0], bounds[1]+bounds[3]), color_key[color], thickness)
    else:
        color = box_data[index-1]["color"]
        bounds = box_data[index-1]["box"]

        # Only live-update the values if a SINGLE box is selected
        sys.stdout.write("\033c")
        print(f"x_border: {x_border}    ")
        print(f"y_border: {y_border}    ")
        print(f"Horizon %: {horizon_percent}    ")
        print(f"cone_scale_factor: {bounds[3]/((bounds[1]+bounds[3])-horizon)}  ")
        print(f"Expected height: {bounds[3]}    ")
        print(f"y_position (bottom) from horizon: {(bounds[1]+bounds[3])-horizon}   ")

        cv.line(input_image, (bounds[0], bounds[1]), (bounds[0]+bounds[2], bounds[1]), color_key[color], thickness)
        cv.line(input_image, (bounds[0], bounds[1]), (bounds[0], bounds[1]+bounds[3]), color_key[color], thickness)

        # bottom right corner
        cv.line(input_image, (bounds[0]+bounds[2], bounds[1]+bounds[3]), (bounds[0]+bounds[2], bounds[1]), color_key[color], thickness)
        cv.line(input_image, (bounds[0]+bounds[2], bounds[1]+bounds[3]), (bounds[0], bounds[1]+bounds[3]), color_key[color], thickness)
    return (input_image)

def make_folders(root: str):
    subfolders = [
        "\\input", 
        "\\input\\img", 
        "\\output",
        "\\output\\img",
        "\\output\\ann",
    ]
    # if the folder does not exist, create it
    # Loop through all folder directories that need to be made
    for folder in subfolders:
        if (not os.path.exists(f"{root}{folder}")):
            os.mkdir(f"{root}{folder}")

def process_frame(frame: np.ndarray, constants: list, box_index: int):
    cone_data = []
    for cone_type in constants:
        # Get the necessary data for each cone type
        color = cone_type["color"]
        cone_scale_factor = cone_type["cone_scale_factor"]
        size = cone_type["size"]
        hue_scale = cone_type["hue_scale"]
        sat_scale = cone_type["sat_scale"]
        val_scale = cone_type["val_scale"]
        lower_bound = cone_type["lower_bound"]
        upper_bound = cone_type["upper_bound"]

        # Process the specific color
        mask = get_mask(frame, hue_scale, sat_scale, val_scale, lower_bound, upper_bound)
        contour = find_contours(mask)

        # Use constants to get the data of each cone
        cone_data += get_cone_boxes(contour, horizon, scale_factor_error, cone_scale_factor, color, size)

    # draw boxes
    draw_boxes(frame, cone_data, box_index)
    return (frame, cone_data)


# clear terminal
os.system('cls||clear')
file_name = input("Please enter the NAME AND EXTENSION of the file you wish to read: ")

# Create the window with the filename
cv.namedWindow(file_name)

# set our error to something large so that most cones get detected
scale_factor_error = 10



# Put the sliders into the window
cv.createTrackbar('horizon (%)', file_name, 0, 1000, nothing)
cv.createTrackbar('box_index', file_name, 0, 200, nothing)
cv.createTrackbar('border_x', file_name, 0, 3000, nothing)
cv.createTrackbar('border_y', file_name, 0, 2000, nothing)

# Make sure output folder exists
root = os.getcwd()
make_folders(root)

# get the image path
input_folder = rf"input\img"
output_root = rf"output"
input_file_path = f"{input_folder}\\{file_name}"
output_image_path = f"{output_root}\\img\\{file_name}"
output_data_path = f"{output_root}\\ann\\{file_name}"
cap = cv.VideoCapture(input_file_path)

with open("constants.json", "r", encoding="utf-8") as file:
    constants = json.load(file)

# Verify the file opened properly
if not cap.isOpened():
    input("Error: Could not open video. Press enter to end the program. ")
else:
    while True:
        # Read the next frame
        ret, frame = cap.read()
        # Stop the loop if there are no frames left (end of video)
        if not ret:
            break
        key_pressed = cv.waitKey(1)
        # Convert image to bgr color
        frame = cv.cvtColor(frame, cv.IMREAD_COLOR_BGR)
        # get frame dimensions
        frame_height, frame_width, channels = frame.shape

        # set bounds for trackbars
        cv.setTrackbarMax('border_x', file_name, round(frame_width/2))
        cv.setTrackbarMax('border_y', file_name, round(frame_height/2))
        
        # reset box index to 0 so that we don't have an indexing problem from previous frame
        cv.setTrackbarPos("box_index", file_name, 0)
        while key_pressed == -1:
            # Get a fresh frame
            line = frame.copy()

            # get slider positions
            horizon_percent = cv.getTrackbarPos('horizon (%)', file_name)/1000
            box_index = cv.getTrackbarPos('box_index', file_name)
            x_border = np.clip(cv.getTrackbarPos('border_x', file_name), 0, round(frame_width/2))
            y_border = np.clip(cv.getTrackbarPos('border_y', file_name), 0, round(frame_height/2))

            horizon = y_border + round((frame_height-(2*y_border))*horizon_percent)
            
            # Process the single frame
            line, frame_data = process_frame(line, constants, box_index)

            # Draw borders
            # y_border
            line = cv.line(line, (x_border, y_border), (frame_width-x_border, y_border), (0, 0, 255), 1)
            line = cv.line(line, (x_border, frame_height-y_border), (frame_width-x_border, frame_height-y_border), (0, 0, 255), 1)

            # x_border
            line = cv.line(line, (x_border, y_border), (x_border, frame_height-y_border), (0, 0, 255), 1)
            line = cv.line(line, (frame_width - x_border, y_border), (frame_width-x_border, frame_height-y_border), (0, 0, 255), 1)

            # horizon
            line = cv.line(line, (x_border, horizon), (frame_width-x_border, horizon), (255, 0, 255), 1)
            
            # Display the frame in a window
            cv.imshow(file_name, line)


            # Check periodically to see if any key is pressed.
            key_pressed = cv.waitKey(1)