import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

FEATURE_TRACK_DATASET = 'gif_tracks/boxes_rotation_198_278_hc.gt.txt'
IMG_DIR = 'ec_data/boxes_rotation_198_278/images/'
GIF_DIR = 'gifs/boxes_rotation_198_278_hc/'

tracks_array = np.loadtxt(FEATURE_TRACK_DATASET)
# print(tracks_array.shape)

tracks = [[] for _ in range(len(np.unique(tracks_array[:, 0])))]

for curr_track in range(tracks_array.shape[0]):
    line = tracks_array[curr_track]
    i = line[0]
    x = line[2]
    y = line[3]
    tracks[int(i)].append((x, y))

prev1s = [None for _ in range(len(tracks))]
prev2s = [None for _ in range(len(tracks))]

# get frames
frame_paths = []
for i, img in enumerate(os.listdir(IMG_DIR)):
    frame_paths.append(os.path.join(IMG_DIR, img))

for t in range(len(tracks[0])):
  frame = cv2.imread(frame_paths[t])

  for track_i in range(len(tracks)):
    x, y = tracks[track_i][t]
    cv2.circle(frame, (int(x), int(y)), radius=3, color=(0, 255, 0), thickness=-1)

    prev1 = prev1s[track_i]
    prev2 = prev2s[track_i]

    if prev1:
      cv2.line(frame, (int(prev1[0]), int(prev1[1])), (int(x), int(y)), (0, 255, 255), 1)
    if prev2:
      cv2.line(frame, (int(prev2[0]), int(prev2[1])), (int(prev1[0]), int(prev1[1])), (0, 175, 175), 1)

    prev2s[track_i] = prev1
    prev1s[track_i] = (x, y)

  plt.imshow(frame)
  cv2.imwrite(os.path.join(GIF_DIR, f"frame_{t}.png"), frame)
#   plt.show()
