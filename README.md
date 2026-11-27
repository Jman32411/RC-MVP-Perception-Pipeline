# RC MVP Perception Pipeline

## Overview
- The goal was to design a vision pipeline that can detect blue and yellow cones.
- Once detected, the program should visually draw a box around them as well as output the color, position, and dimension data in the format
`[{color, box: [x, y, w, h]}]`.
- Must run at 10 FPS or more at a resolution of 640x480 or greater.
## Installation
To install this project, make sure you have at least python 3.14 and pip 26.3
- Because this project is a collection of python scripts, it can be downloaded as just a .zip file.
- Once extracted to a desired location, ensure all of the necessary dependencies are installed by running the following command in the project folder.
``` bash
pip install -r requirements.txt
```
## File structure
The main file-structure of this project is fairly simple.  
***ROOT***  
├── input/  
│&emsp;&emsp;└── img/  
│&emsp;&emsp;&emsp;&emsp;&emsp;├── video_name.mp4  
│&emsp;&emsp;&emsp;&emsp;&emsp;└── image_name.png  
├── output/  
│&emsp;&emsp;├── ann/  
│&emsp;&emsp; |&emsp;&emsp;&emsp;├── video_name.mp4.json   
│&emsp;&emsp; |&emsp;&emsp;&emsp;└── image_name.png.json  
│&emsp;&emsp;└── img/  
├── testing_scripts/  
├── box_drawing_v3.py  
├── parameter_finder.py  
├── constants.json  
├── parameters.json  
... other files  

### Input Files
- This project ***should*** be able to handle any form of visual media input (video and image).
- By placing all input files into the "input\img" folder, it should be able to be able to be read properly from the program

### Output Files
- If the programs have run properly, then the outputs should only be in the "output\ann" folder.
    - The reason the "img" folder is in the output folder is to allow for the future possibility of saving the marked images and videos
- The output name should match the input file name, just with a .json extension.


## How to use
### Setting the parameters
1. Once your desired file has been placed into the "input\img" folder, run the "parameter_finder.py" program.
2. It should prompt you to enter a file name. Make sure you ONLY put the name of the file (including file extension) into the prompt.
3. Once the image or video has loaded, use the sliders to adjust x and y borders (highlighted in red) to capture only the desired area.
4. Then, use the horizon % slider to adjust the line (magenta) until it sits where the horizon would be.
5. If there is a fully highlighted cone, move the box index slider until just that cone is highlighted.
6. If the file you are viewing is a video and the current frame either does not have a cone that is properly highlighted or if you wish to view a different frame, you can press any key to advance to the next frame.
7. Once all of the parameters are set, look at the terminal and note down all 4 values. 
8. Using these values, go into the "parameters.json" file, and replace the corresponding values with the ones found before.

### Drawing the boxes
1. Once the parameters have been found and set, run the "box_drawing_v3.py" file.
2. It will, just like "parameter_finder.py", ask for a file name. Enter ONLY the name of the file (including file extension).
3. If everything goes well, the program should progress through every frame (if it's a video), marking a box around each detection. When it reaches the final frame, or if it is just an image, the program will freeze until any key is pressed.
4. Every detection for each frame will be listed in the .json file with the corresponding name in "output\ann".


## How it works
### box_drawing_v3.py
1. The program starts by getting the file that it is supposed to analyze.
2. It also pulls the parameters from "parameters.json" and the constants from "constants.json"
3. Using the file-name, it opens the file and if it is a video, reads the first frame.
4. For every frame, it converts the image into a 3-color BGR channel, gets the frame dimensions, and calculates the horizon line position (using data pulled from parameters).
5. It then processes the given frame for each cone type listed in constants.
    1. It will normalize the hue (grey-world normalization), saturation, and value to a specific range (averaged, then boosted/or reduced based on previous testing).
    2. It will then generate a binary mask of pixels that fall within a certain color range (again defined within constants).
    3. Using this binary mask, the program then finds the contours, creates a box around them, and filters them based on size, shape/aspect ratio, and % color filled. 
    4. It then looks for boxes that are within a certain distance above other boxes, and sees if it can join them together. This handles the fact that the cones are separated into 2 parts due to the central strip
    5. Once joined, the boxes are then filtered again based on size and distance from the horizon line.
6. This information is then added to an aggregate dataset.
7. This information is also used to draw, on the displayed frame, the bounding boxes.
8. When all of the boxes are drawn, the next frame is displayed and the process repeats.
9. When the program is finally through each frame, the data is then written to the output file.

### Parameter_finder.
Using "box_drawing_v3.py" as a skeleton, the parameter finder was updated to include sliders to adjust the bounds/borders of the frame, as well as easilly draw and find where the horizon line falls. The sliders also were used to help isolate a specific cone to calculate the scaling factor. By pulling the relevant information, the necessary data can be calculated and output to the terminal. Very little got changed from "box_drawing_v3.py", just the fact that parameters can be tweaked on-the-fly using the sliders, which then can be used by the actual box drawing program to (hopefully) get better results.

## Issues and Future Work
- ***__Accuracy/Noise__***
    - As seen by just running the program, there are multiple instances where background noise gets interpreted as a cone.
    - Additionally, there are instances where overlapping cones, or certain cones get discarded due to their odd size. 
    - This could be fixed with better filtering/masking, but I think that by using a Neural Network, this could be completely removed.
- ***__Lighting/Color__***
    - While this is similar to Accuracy/Noise, the fact that this is an outdoor competition means that the lighting conditions will vary widely.
    - I think the idea of having a reference/normalization patch/grid always in frame could help with camera calibration and adjustments to the light.
- ***__Performance__***
    - The program, as is, barely runs at 12 FPS with a [1120x480 downscaled video](https://www.youtube.com/watch?v=o5vES5QaeiQ)
    - One of the main culprits of this terrible performance is the fact that it takes ***TWO*** loops to fully process a single frame (one to get the contours and run an intial filter and another to do the vertical-box joining and a secondary filter)
    
## Documentation/Process
If you want to just look at my raw-unfiltered thoughts throughout the project, you can look at my Google Document note-sheet [here](https://docs.google.com/document/d/113G8CNeRWPbQwE5lvHY49GNGjqd8b962rAE1FGdqTR8/edit?usp=sharing). If not, you can read a ***MUCH MORE*** condensed version below.

### Initial Research
- Looked into Python systems, as that is the language I am most comfortable with
- Some options that came up include:
    - OpenCV
    - Pillow
    - scikit-image
    - NumPy + Numba
    - NCNN
    - Neural Networks (YOLO and/or Torch)
- Looked into the [dataset](https://fsoco.github.io/fsoco-dataset/download) and found a [sample dataset](https://drive.google.com/file/d/1l2k7q0KG7ejqquepgJBMeJkwueVutf5t/view)
    - Uses the Supervisely image annotation format
    - Has a bounding box and panoptic segmentation dataset
- Leaned away from using Neural Networks for 3 reasons:
    1. Unsure about performance on such a lightweight system
    2. I have/had a feeling that other people would try to use a neural network program (YOLO), as Supervisely has integrated tools to do that.
    3. I wanted to see if it was possible to come up with something that doesn't require neural network training.

### Initial Data Review
- Upon downloading the sample dataset I had multiple thoughts and ideas
    - How can I utilize Supervisely to more effectively analyze this data?
    - The data has such a wide range of lighting conditions and resolutions
- Researched how I can utilize YOLO inside the Supervisely ecosystem
- Researched how to  normalize the varying lighting conditions.

### Initial Data Findings
- With an environment that has such dynamic lighting, you cannot just apply a single image mask based on a color range.
- Hue, Saturation, and Value normalization could help counteract the lighting.
    - A Grey-world normalization was used for the hue, then adapted to average the hue and saturation. 
- Additionally, size filtering could be used to help reduce noise
- ***Ideas for the prod car*** include:
    - A normalization grid/patch on our car.  A grid with a fixed size parts consisting of different, known color values (like pure red, green, blue, black, and white).
- YOLO training COULD work, but might be too intensive and black-box ish.

### Image and Box Filters
- Attempted to implement multiple image and color filters.
- Started working on finding HSV normalization/boosts that could help isolate the yellow cones.
- Additionally, when detecting the cones based on color, most cones are split into 2 segments.
- To fix this, I brainstormed ideas to look at small boxes, extrapolate downwards, and see if it falls within the bounds of a larger cone. If so, join the two together. (No code was written, just brainstormed)
- For background noise, just filter by box size, but after testing, filtering just based on size would not work, as farther cones might get mistaken for nosie (and vice versa).
- To try and fix this I came up with 3 things:
    - Filter by aspect ratio (remove extremely thin or fat boxes)
    - Filter by contour comparison (area of contour vs area of rectangle)
    - Filter based on distance from horizon
- To fix the lighting conditions, I looked into CLAHE, but it was either not working effectively enough, or I was implementing it incorrectly (most likely this)
- Looked to implement a height-based filter using a simple linear scaling factor: $a = E_{xpectedHeight}/D_{istanceFromHorizon}$
