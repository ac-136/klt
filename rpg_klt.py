import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from PIL import Image 
from feature_point import harris_corner
import random

random.seed(42)

def trackPoints(xy, imageSequence, times):
  print ("In function trackPoints")
  print (f'length of imageSequence = {len(imageSequence)}')
  # movedOutFlag = np.zeros(xy.shape[0])
  # initialize xyt to contain any information that is needed for drawing paths at the end of tracking
  # also add code in this function as needed to maintain xyt
  #xyt = 0

  # xyt is initialized as a list, to which the new points from getNextPoints can be added
  xyt = []
  for index, arr in enumerate(xy):
    new_arr = np.insert(arr, 0, index)
    new_arr = np.insert(new_arr, 1, times[0])
    xyt.append(new_arr)

    xy[index] = xy[index].astype(np.float32)

  xy = np.array(xy)
  xy = np.expand_dims(xy, axis=1).astype(np.float32)
  print(xy.shape)

  for t in range(0, len(imageSequence)-1): # predict for all images except first in sequence
    # print (f't = {t}; predicting for t = {t+1}') 
    # xy2, status, err = cv2.calcOpticalFlowPyrLK(imageSequence[t], imageSequence[t+1], xy,
    #                                      None, winSize=(21, 21), maxLevel=1)
    xy2, status, err = cv2.calcOpticalFlowPyrLK(imageSequence[t], imageSequence[t+1], xy,
                                         None, winSize=(15, 15), maxLevel=2)
    xy = xy2

    new_xy2 = []
    for index, arr in enumerate(xy2):
        new_arr = np.insert(arr, 0, index)
        new_arr = np.insert(new_arr, 1, times[t+1])
        new_xy2.append(new_arr)

    for pt in new_xy2:
      xyt.append(pt)
    
  return xyt


# DIR = "ec_data/shapes_translation/images/"
# DIR_TIME = "ec_data/shapes_translation/times.txt"
# OUTPUT_DIR = "feature_tracks/shapes_translation_8_88.gt.txt"

ROOT = "ec_data"

data_list = []
for i, data in enumerate(os.listdir(ROOT)):
    data_list.append(data)

for d in data_list:
    DIR = os.path.join(ROOT, d, "images")
    DIR_TIME = os.path.join(ROOT, d, "times.txt")
    OUTPUT_DIR = os.path.join("feature_tracks", d + ".gt.txt")

    ##### Create list of images for dataset #####
    imgs_list = []
    for i, img in enumerate(os.listdir(DIR)):
        curr_path = os.path.join(DIR, img)
        curr_img = cv2.imread(curr_path, cv2.COLOR_BGR2GRAY)
        imgs_list.append(curr_img)

    # Number of Images (should always be 81)
    print("Num images:", len(imgs_list))

    ##### Get list of associated times #####
    with open(DIR_TIME, 'r') as file:
        # Read the lines and store them in a list
        times = file.readlines()
    print("Num times:", len(times))

    ##### Get initial keypoints #####
    img_0 = imgs_list[0]
    kp_xy, im0color = harris_corner(img_0)

    ##### Track keypoints over remaining images #####
    xyt = trackPoints(kp_xy, imgs_list, times)

    # sorted_xyt = sorted(xyt, key=lambda arr: arr[0])

    # print(len(sorted_xyt))
    # print(sorted_xyt[0])
    # print(sorted_xyt[81])
    # print(sorted_xyt[162])

    ##### Write to .gt.txt #####
    with open(OUTPUT_DIR, 'w') as file:
        # Iterate over the list of arrays
        for array in xyt:
            # Convert the array elements to strings and join them with spaces
            array[0] = int(array[0])
            row = ' '.join(map(str, array))
            # Write the row to the file
            file.write(row + '\n')