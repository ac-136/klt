# Overview
### used python 3.7.9

# Code
## klt.py
* tracking of feature found using different feature detectors is adapted from this code: https://github.com/nirmal-25/Kanade-Lucas-Tomasi-KLT-feature-tracker
* different feature tracking detectors used in feature_point.py

## feature_point.py
* code for different feature detectors including: harris corner, sift, fast, and orb

## pose_klt.py
* uses feature tracks from klt.py using harris corner detector
* triangulates feature tracks from 2D to 3D points based off camera parameters and pose
* reprojects these 3D points for each frame
* based off this code: https://github.com/uzh-rpg/deep_ev_tracker/blob/608655b997c6d58ae25a29d833551509888b4349/utils/track_utils.py

## rpg_klt.py
* uses klt feature tracking that the authors have used in the past
* based on this code: https://github.com/uzh-rpg/rpg_feature_tracking_analysis/blob/af8496eaea5df432a008e67027b1757b81a0405b/big_pun/tracker.py

## gen_gif.py
* draws ground truth feature tracks over frames

## shorten_ec_data.py
* makes sure the ec data is in the correct format for deep_ev_tracker

# Data/Folders
### Original EC dataset not included because its around 7.5 GB

## ec_data
* correctly formated ec data for the deep_ev_tracker
* generated using shorten_ec_data.py

## feature_tracks_
* ground truth feature tracks generated using different feature tracking methods
* hc = harris-corner, pose = harris-corner + poser, sift = sift, fast = fast, orb = orb
* serves as input ground truth truth for deep_ev_tracker

## gif_tracks
* ground truth feature tracks from different feature_tracks folders to generate feature tracks over the frames

## gifs
* ground truth feature tracks over frames
* generated using gen_gif.py