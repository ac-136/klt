import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from PIL import Image 
from feature_point import harris_corner, find_sift, find_fast, find_orb
import random

random.seed(42)
DISPLAY_RADIUS = 3
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)

def linear_filter(img, kernel):

  img_x, img_y = img.shape
  kernel_x, kernel_y = kernel.shape
  kernel_x = int(np.floor(kernel_x/2))
  kernel_y = int(np.floor(kernel_y/2))

  img_padded = np.pad(img, (kernel_x, kernel_y), 'constant', constant_values=0)
  X, Y = img_padded.shape

  conv_output = np.zeros((img_x, img_y))

  for i in range(kernel_x, X-kernel_x):
      for j in range(kernel_y, Y-kernel_y):
          window = img_padded[i-kernel_x:i+kernel_x+1, j-kernel_y:j+kernel_y+1]
          conv_output[i-kernel_x][j-kernel_y] = np.sum(window*kernel)

  return conv_output

def getNextPoints(im1, im2, xy, movedOutFlag):
#   print("In function getNextPoints")

  xy2 = np.copy(xy).astype(float)
  im1 = im1.astype(np.float32)
  im2 = im2.astype(np.float32)

  # Gaussian kernel for smoothing
  kernel = np.array([
          [1, 4, 7, 4, 1],
          [4, 16, 26, 16, 4],
          [7, 26, 41, 26, 7],
          [4, 16, 26, 16, 4],
          [1, 4, 7, 4, 1,]], dtype=np.float32) / 273.0

  img = linear_filter(im1, kernel).astype(np.float32)
  Iy, Ix =  np.gradient(img)

  # The given KLT algorithm is implemented
  for i in range(len(xy)):
    patch_x = cv2.getRectSubPix(Ix, (15,15), (xy[i,0], xy[i,1]))
    patch_y = cv2.getRectSubPix(Iy, (15,15), (xy[i,0], xy[i,1]))
    A = np.array([[np.sum(patch_x * patch_x), np.sum(patch_x * patch_y)], [np.sum(patch_x * patch_y), np.sum(patch_y * patch_y)]])

    for j in range(25):
      patch_t = cv2.getRectSubPix(im2, (15,15), (xy2[i,0], xy2[i,1])) - cv2.getRectSubPix(img, (15, 15), (xy[i,0], xy[i,1]))
      B = -1* np.array([[np.sum(patch_x*patch_t)],[np.sum(patch_y*patch_t)]])
      disp = np.matmul(np.linalg.pinv(A), B)

      u = disp[0]
      v = disp[1]

      xy2[i] = [xy2[i,0] + u, xy2[i,1] + v] 

      # Checking if the norm of (u, v) is leser than the threshold (from the textbook section - 9.1.3 Incremental refinement)
      if np.hypot(u, v) <= 0.01:
        break

  # Setting the movedOutFlag to 1 if the new pixels are out of bounds      
  if xy2[i,0] >= len(im1) or xy2[i,0] < 0 or xy2[i,1] >= len(im1[0]) or xy2[i,1] < 0:
    movedOutFlag[i] = 1

  return(xy2, movedOutFlag)

def trackPoints(xy, imageSequence, times):
  print ("In function trackPoints")
  print (f'length of imageSequence = {len(imageSequence)}')
  movedOutFlag = np.zeros(xy.shape[0])
  # initialize xyt to contain any information that is needed for drawing paths at the end of tracking
  # also add code in this function as needed to maintain xyt
  #xyt = 0

  # xyt is initialized as a list, to which the new points from getNextPoints can be added
  xyt = []
  for index, arr in enumerate(xy):
    new_arr = np.insert(arr, 0, index)
    new_arr = np.insert(new_arr, 1, times[0])
    xyt.append(new_arr)

  for t in range(0, len(imageSequence)-1): # predict for all images except first in sequence
    # print (f't = {t}; predicting for t = {t+1}') 
    xy2, movedOutFlag = getNextPoints(imageSequence[t], imageSequence[t+1], xy, movedOutFlag)
    xy = xy2

    new_xy2 = []
    for index, arr in enumerate(xy2):
        new_arr = np.insert(arr, 0, index)
        new_arr = np.insert(new_arr, 1, times[t+1])
        new_xy2.append(new_arr)

    for pt in new_xy2:
      xyt.append(pt)

    # # for selected instants in time, display the latest image with highlighted keypoints 
    # if ((t == 0) or (t == 10) or (t == 20) or (t == 30) or (t == 40) or (t == 49)):
    #   im2color = cv2.cvtColor(imageSequence[t+1], cv2.COLOR_GRAY2BGR)
    #   corners = np.intp(np.round(xy2))
        
    #   for c in range(0, corners.shape[0]):
    #     if movedOutFlag[c] == False:
    #       x = corners[c][0]
    #       y = corners[c][1]
    #       cv2.circle(im2color, (x, y), DISPLAY_RADIUS, GREEN)
    #   plt.imshow(im2color)
    #   plt.show()
    
  return xyt

def drawPaths(im0color, xyt):
  print ("In function drawPaths")

  # Using cv2.circle to draw dots for all new points in xyt, by setting radius to 0
  for pt in xyt:
    im0color = cv2.circle(im0color, (round(pt[1]),round(pt[2])), radius=0, color=YELLOW, thickness=1)

  print ("FINISHED: here are the paths of the tracked keypoints")
  plt.imshow(im0color)
  plt.show()

# boxes_rotation, boxes_rotation_198_278
# boxes_translation, boxes_translation_330_410
# shapes_6dof, shapes_6dof_485_565
# shapes_rotation, shapes_rotation_165_245
# shapes_translation, shapes_translation_8_88

ROOT = "ec_data"

data_list = []
for i, data in enumerate(os.listdir(ROOT)):
    data_list.append(data)

for d in data_list:
    DIR = os.path.join(ROOT, d, "images")
    DIR_TIME = os.path.join(ROOT, d, "times.txt")
    OUTPUT_DIR = os.path.join("feature_tracks", d + ".gt.txt")
    GIF_DIR = os.path.join("gifs/", d)

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
    # kp_xy, im0color = find_sift(img_0)
    # kp_xy, im0color = find_fast(img_0)
    # kp_xy, im0color = find_orb(img_0)

    #### Track keypoints over remaining images #####
    xyt = trackPoints(kp_xy, imgs_list, times)

    sorted_xyt = sorted(xyt, key=lambda arr: arr[0])

    # drawPaths(im0color, sorted_xyt)

    ##### Write to .gt.txt #####
    with open(OUTPUT_DIR, 'w') as file:
        # Iterate over the list of arrays
        for array in xyt:
            # Convert the array elements to strings and join them with spaces
            array[0] = int(array[0])
            row = ' '.join(map(str, array))
            # Write the row to the file
            file.write(row + '\n')



