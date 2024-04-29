import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from PIL import Image 

DISPLAY_RADIUS = 3
DISPLAY_COLOR  = (0, 255, 0)

def harris_corner(im0):
  # find image locations that may be good for tracking
  feature_params = dict( maxCorners = 30,
                       qualityLevel = 0.2, 
                       minDistance = 25,  
                       blockSize = 25 )     
  # feature_params = dict( maxCorners = 100,
  #                     qualityLevel = 0.2, 
  #                     minDistance = 7,  
  #                     blockSize = 5 )   
  # feature_params = dict( maxCorners = 100,
  #                     qualityLevel = 0.3, 
  #                     minDistance = 30,  
  #                     blockSize = 25 )  
  p0 = cv2.goodFeaturesToTrack(im0, mask = None, **feature_params)
  # now corners should contain an array of (floating point) pixel locations 
  if p0 is None:
    print("no keypoints were found!")
    return
  print (f'Number of detected keypoints = {p0.shape[0]}')

  # convert to kx2 format, where k is the number of feature points
  corners = np.zeros((p0.shape[0],2))
  for i in range(corners.shape[0]):
    corners[i] = p0[i][0]

  # draw a small circle at each detected point and display the result
  im0color = cv2.cvtColor(im0, cv2.COLOR_GRAY2BGR)
  cornersInt = np.intp(np.round(corners)) # convert to integers used for indexing 
  for i in cornersInt:
    x, y = i.ravel()      # returns a contiguous flattened array
    cv2.circle(im0color, (x, y), DISPLAY_RADIUS, DISPLAY_COLOR)

#   plt.imshow(im0color)
#   plt.show()
  return corners, im0color

def find_sift(im0):
  sift = cv2.SIFT_create()

  sift.setContrastThreshold(0.105)
  sift.setEdgeThreshold(3)

  kp = sift.detect(im0, None)

  if kp is None:
    print("no keypoints were found!")
    return
  print (f'Number of detected keypoints = {len(kp)}')
  
  # convert to kx2 format, where k is the number of feature points
  corners = np.zeros((len(kp),2))
  for i in range(len(kp)):
    corners[i][0] = kp[i].pt[0]
    corners[i][1] = kp[i].pt[1]

  # draw a small circle at each detected point and display the result
  im0color = cv2.cvtColor(im0, cv2.COLOR_GRAY2BGR)
  cornersInt = np.intp(np.round(corners)) # convert to integers used for indexing 
  for i in cornersInt:
    x, y = i.ravel()      # returns a contiguous flattened array
    cv2.circle(im0color, (x, y), DISPLAY_RADIUS, DISPLAY_COLOR)

  # plt.imshow(im0color)
  # plt.show()
  # return corners, im0color

# def fast():

# def orb():

