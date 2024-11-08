import os
import numpy as np
from soft_manipulator_curve_fitting import get_task_pose
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from utils import inverse_kinematics, compute_task_error
from segment_merging_algorithm import segment_merging_algorithm

######## Define initial parameters ###########
num_segments = 1
# params = {"l": 1e-1 * np.ones((num_segments,))}
params = {"l": 0.15 * np.ones((num_segments,))}
# params = {"l": np.array([0.05, 0.1, 0.06])}
# params = {"l": np.array([0.07, 0.1])}
params["total_length"] = np.sum(params["l"])
# params = {"l": np.array([1e-1, 5e-2])}
video_height = 2360
video_height = 2860
# video_height = 4360
# ppm = video_height / (1.8 * np.sum(params["l"]))
ppm = video_height / (2.3 * np.sum(params["l"]))
# small number to avoid singularities on IK and FK
eps = 1e-7
# threshold for merging segments
# threshold = 0.068#0.064
threshold = 0.2
##############################################
# string of strains for plotting
string_strains = ['Bending', 'Shear', 'Axial']

# STEP 1: Check videos and actuation
videos_folder = f"videos/ns-{num_segments}_high_shear_stiffness/"
videos_folder = f"videos/ns-{num_segments}_noise_larger/"
videos_folder = f"videos/ns-{num_segments}_end-to-end/"
videos_folder = f"videos/ns-{num_segments}_video_ppt/"
videos_folder = f"videos/ns-{num_segments}_pac_video_ppt/"
# videos_folder = f"videos/ns-{num_segments}_homogeneous/"
# videos_folder = f"videos/ns-{num_segments}_test/"
# videos_folder = f"videos/ns-{num_segments}_pac/"
# videos_folder = f"videos/ns-{num_segments}_dof-3_high_shear_stiffness/"
# videos_folder = f"videos/ns-{num_segments}_dof-2_bending_shear/"
# videos_folder = f"videos/ns-{num_segments}_dof-3_stiff_shear_and_torques/"
video_names = [name for name in os.listdir(videos_folder) if os.path.isfile(os.path.join(videos_folder, name))]

# Compute kinematic error metrics and generate video for test trajectory
# STEP 1: Get the configurations assuming the determined Kinematic model
pose_data = get_task_pose(videos_folder + video_names[-1], ppm)
pose_data[:,:,2] = pose_data[:,:,2]*np.pi/180 # convert deg to rad
num_cs = pose_data.shape[1]
s = params["total_length"] / (num_cs - 1)
seg_length = s*np.ones((num_cs - 1))
config_data = inverse_kinematics(pose_data, eps, s, plot=True)
config_data_list = [config_data]
s_image_cum = np.cumsum(seg_length) # all the points we track
T = pose_data.shape[0] # number of frames

# STEP 2: Define the merged segment lengths after the Kinematic Regression
merged_segment_lengths = [seg_length, np.array([0.0075, 0.105, 0.0375])]
# cumsum of the segment lengths
s_cum = np.cumsum(merged_segment_lengths[1])
# add zero to the beginning of the array
s_cum_padded = np.concatenate([np.array([0.0]), s_cum], axis=0, dtype=np.float32)

# STEP 3: Compute the new configurations after the merging
pose_previous_frame = np.zeros((T,3))
N = len(s_cum)

config = np.zeros((T,N,3))
for id_seg, s_point in enumerate(s_cum):
    # determine the cross-section correspondent to the end of the segment
    cs_idx = np.where(np.isclose(s_image_cum, s_point))[0] + 1
    s = np.diff(s_cum_padded)[id_seg]

    pose_current_frame = pose_data[:,cs_idx,:].reshape((T,3))

    relative_pose_data = pose_current_frame - pose_previous_frame
    for i in range(T):
        theta = pose_previous_frame[i,2]
        relative_pose_data[i,:2] = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]) @ relative_pose_data[i,:2]
    
    px = relative_pose_data[:,0]
    py = relative_pose_data[:,1]
    theta = relative_pose_data[:,2]

    # add small eps for numerical stability
    th_sign = np.sign(theta)
    # set zero sign to 1 (i.e. positive)
    th_sign = np.where(th_sign == 0, 1, th_sign)
    # add eps to theta
    # theta_eps = theta + th_sign * eps
    theta_eps = np.select(
        [np.abs(theta) < eps, np.abs(theta) >= eps],
        [th_sign*eps, theta]
    )

    config[:,id_seg,0] = theta_eps / s
    config[:,id_seg,1] = ( theta_eps / (2*s) ) * (py - (px*np.sin(theta_eps))/(np.cos(theta_eps) - 1)) # shear strain
    config[:,id_seg,2] = ( theta_eps / (2*s) ) * (-px - (py*np.sin(theta_eps))/(np.cos(theta_eps) - 1)) # axial strain
    config[:,id_seg,2] = config[:,id_seg,2] - 1

    pose_previous_frame = pose_current_frame

config_data_list.append(config)

pose_data_iterations, error_metric_iterations = compute_task_error(pose_data, config_data_list, merged_segment_lengths, eps)



