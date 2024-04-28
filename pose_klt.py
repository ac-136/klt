import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from PIL import Image 
from feature_point import harris_corner
import random

from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp

random.seed(42)

# based off data driven github using the following files/parts of files: 
# https://github.com/uzh-rpg/deep_ev_tracker/blob/608655b997c6d58ae25a29d833551509888b4349/scripts/benchmark.py
# https://github.com/uzh-rpg/deep_ev_tracker/blob/608655b997c6d58ae25a29d833551509888b4349/utils/track_utils.py#L339
# https://github.com/uzh-rpg/deep_ev_tracker/blob/608655b997c6d58ae25a29d833551509888b4349/utils/track_utils.py#L471
# https://github.com/uzh-rpg/deep_ev_tracker/blob/608655b997c6d58ae25a29d833551509888b4349/utils/track_utils.py#L174
# https://github.com/uzh-rpg/deep_ev_tracker/blob/608655b997c6d58ae25a29d833551509888b4349/utils/track_utils.py#L283
# https://github.com/uzh-rpg/deep_ev_tracker/blob/608655b997c6d58ae25a29d833551509888b4349/utils/track_utils.py#L237


##### triangulate and reproject all feature tracks #####
FEATURE_TRACK_DIR = 'feature_tracks'
OUTPUT_DIR = 'pose_feature_tracks/'

for i, ft in enumerate(os.listdir(FEATURE_TRACK_DIR)):
    curr_path = os.path.join(FEATURE_TRACK_DIR, ft)

    curr_dataset = []
    with open(curr_path, 'r') as ft_file:
        for line in ft_file:
            values = line.split()
            curr_dataset.append([float(val) for val in values])

    curr_dataset = np.array(curr_dataset)
    
    EC_DIR = 'ec_data_og/'
    path_parts = curr_path.split('\\', 3)
    name_parts = path_parts[1].split('_', 2)
    curr_name = name_parts[0] + '_' + name_parts[1]
    
    CURR_EC_DIR = EC_DIR + curr_name

    ### get camera matrix ###
    CAMERA_DIR = CURR_EC_DIR + "/calib.txt"
    cam_matrix = np.zeros((3,3))
    # camera parameters = [fx fy cx cy k1 k2 p1 p2 k3]
    with open(CAMERA_DIR, 'r') as cam_file:
        for line in cam_file:
            values = line.split()

            cam_matrix[0, 0] = values[0]
            cam_matrix[0, 2] = values[2]
            cam_matrix[1, 1] = values[1]
            cam_matrix[1, 2] = values[3]

            cam_matrix[2, 2] = 1

    ### get poses as projection matrix ###
    POSE_DIR = CURR_EC_DIR + '/groundtruth.txt'
    pose_data = []
    # pose data = Nx7 file with [t, x, y, z, qx, qy, qz, qw] for rows
    with open(POSE_DIR, 'r') as pose_file:
        for line in pose_file:
            values = line.split()
            pose_data.append([float(val) for val in values])

    pose_data = np.array(pose_data)
    
    x_interp = interp1d(pose_data[:, 0], pose_data[:, 1], kind='linear', bounds_error=True)
    y_interp = interp1d(pose_data[:, 0], pose_data[:, 2], kind='linear', bounds_error=True)
    z_interp = interp1d(pose_data[:, 0], pose_data[:, 3], kind='linear', bounds_error=True)

    rot_interp = Slerp(pose_data[:, 0], Rotation.from_quat(pose_data[:, 4:]))

    def interpolate(i_t):
        T_W_j = np.eye(4)
        T_W_j[0, 3] = x_interp(i_t)
        T_W_j[1, 3] = y_interp(i_t)
        T_W_j[2, 3] = z_interp(i_t)
        T_W_j[:3, :3] = rot_interp(i_t).as_matrix()
        return np.linalg.inv(T_W_j)
    
    reproj_dataset_ft = np.zeros((0, 4))
    for track_idx in np.unique(curr_dataset[:, 0]):
        curr_feat_track = curr_dataset[curr_dataset[:, 0] == track_idx, 1:]
        
        track_data_with_id = np.concatenate([np.zeros((curr_feat_track.shape[0], 1)), curr_feat_track], axis=1)
        t_init = np.min(track_data_with_id[:, 1])

        # pose interpolation
        T_init_W = interpolate(t_init)

        ### calculate triangulated 3D points ###
        pts_3D_homo = []
        A = []
        track_data_curr = track_data_with_id[:, 1:]

        for idx in range(track_data_curr.shape[0]):
            pt = track_data_curr[idx, 1:]
            t = track_data_curr[idx, 0]

            T_j_W = interpolate(t)
            T_j_init = T_j_W @ np.linalg.inv(T_init_W)

            # projection
            P = cam_matrix @ T_j_init[:3, :]
            A.append(pt[0] * P[2, :] - P[0, :])
            A.append(pt[1] * P[2, :] - P[1, :])

        A = np.array(A)
        _, _, vh = np.linalg.svd(A)
        X = vh[-1, :].reshape((-1))
        X /= X[-1]

        pts_3D_homo = X.reshape((1, 4))

        ### reproject points at each timestep ###
        curr_track_data_reproj = []
        for t in curr_feat_track[:, 0]:
            T_j_W = interpolate(t)
            T_j_init = T_j_W @ np.linalg.inv(T_init_W)

            pts_3D = (T_j_init @ pts_3D_homo.T).T
            pts_3D = pts_3D[:, :3]
            pts_2D_proj = (cam_matrix @ pts_3D.T).T
            pts_2D_proj = pts_2D_proj / pts_2D_proj[:, 2].reshape((-1, 1))
            pts_2D_proj = pts_2D_proj[:, :2]
            feature_reproj = pts_2D_proj.reshape((-1,))

            curr_track_data_reproj.append([t, feature_reproj[0], feature_reproj[1]])
        
        curr_track_data_reproj = np.array(curr_track_data_reproj)
        id_col = np.full((curr_track_data_reproj.shape[0], 1), track_idx)
        curr_tracks = np.hstack((id_col, curr_track_data_reproj))

        reproj_dataset_ft = np.vstack((reproj_dataset_ft, curr_tracks))

    # write to folder
    curr_out = OUTPUT_DIR + ft
    with open(curr_out, 'w') as outfile:
        for array in reproj_dataset_ft:
            array[0] = int(array[0])
            row = ' '.join(map(str, array))
            # Write the row to the file
            outfile.write(row + '\n')
