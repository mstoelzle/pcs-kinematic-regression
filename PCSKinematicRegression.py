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
video_height = 2360
# video_height = 4360
# ppm = video_height / (1.8 * np.sum(params["l"]))
ppm = video_height / (1.8 * np.sum(params["l"]))
# small number to avoid singularities on IK and FK
eps = 1e-7
# threshold for merging segments
threshold = 0.064
get_configurations = True
##############################################
# string of strains for plotting
string_strains = ['Bending', 'Shear', 'Axial']

# STEP 1: Check videos and actuation
# videos_folder = f"videos/ns-{num_segments}_pics/"
# videos_folder = f"videos/ns-{num_segments}_results/"
videos_folder = f"videos/ns-{num_segments}_pac/"
# videos_folder = f"videos/ns-{num_segments}_dof-3_high_shear_stiffness/"
# videos_folder = f"videos/ns-{num_segments}_dof-2_bending_shear/"
# videos_folder = f"videos/ns-{num_segments}_dof-3_stiff_shear_and_torques/"
video_names = [name for name in os.listdir(videos_folder) if os.path.isfile(os.path.join(videos_folder, name))]

X, Xdot, Tau = [], [], []
for video in video_names:
    print('Video: ' + video + '\n')
    video_path = videos_folder + video
    
    # STEP 2: Get task space pose using CV
    pose_data = get_task_pose(video_path, ppm)
    pose_data[:,:,2] = pose_data[:,:,2]*np.pi/180 # convert deg to rad

    # STEP 3: Convert from cartesian to configuration space
    num_cs = pose_data.shape[1]
    s = params["total_length"] / (num_cs - 1)
    seg_length = s*np.ones((num_cs - 1))
    config_data = inverse_kinematics(pose_data, eps, s, plot=True)

    # STEP 4: Run the segment merging algortihm
    config_data_iterations, seg_length_iterations, _ = segment_merging_algorithm(config_data, seg_length, threshold, filtering=False)

    # STEP 5: Compute kinematic error metrics in task space for each iteration of the algorithm
    pose_data_iterations, error_metric_iterations = compute_task_error(pose_data, config_data_iterations, seg_length_iterations, eps)

    # STEP 6: Smooth out final configurations and compute time derivatives
    if get_configurations:
        T, N_SEG, _ = config_data_iterations[-1].shape
        dt = 1e-3
        q = np.zeros((config_data_iterations[-1].shape))
        for seg in range(N_SEG):
            for strain in range(3):
                if np.max(abs(config_data_iterations[-1][:, seg, strain]), axis=0) > 2e-2:
                    q[:, seg, strain] = savgol_filter(config_data_iterations[-1][:, seg, strain], 5*5, polyorder=3, deriv=0, axis=0)
                else:
                    # q[:, seg, strain] = savgol_filter(config_data_iterations[-1][:, seg, strain], 201, polyorder=2, deriv=0, axis=0)
                    q[:, seg, strain] = savgol_filter(config_data_iterations[-1][:, seg, strain], 5*5, polyorder=3, deriv=0, axis=0)


        # q = savgol_filter(config_data_iterations[-1], 5*5, polyorder=3, deriv=0, axis=0)
        # q_d = savgol_filter(config_data, 6*5 + 1, polyorder=3, deriv=1, delta=dt, axis=0, mode='constant')
        q_d = np.gradient(q, dt, axis=0, edge_order=2)
        q_d[0,:,:] = 0
        q_dd = np.gradient(q_d, dt, axis=0, edge_order=2)
        # q_dd = savgol_filter(config_data, 6*5 + 1, polyorder=3, deriv=2, delta=dt, axis=0)

        # STEP 7: Plot q, q_dot, q_ddot for each strain
        
        time_arr = np.arange(0.0, T*dt, dt)
        for strain in range(3):
            fig, ax = plt.subplots(3,1)
            fig.suptitle(string_strains[strain] + ' data')

            ax[0].grid(True)
            ax[0].set_ylabel('$\ddot{q}$')
            ax[1].grid(True)
            ax[1].set_ylabel('$\dot{q}$')
            ax[2].grid(True)
            ax[2].set_ylabel('$q$')
            for seg in range(N_SEG):
                ax[0].plot(time_arr, q_dd[:, seg, strain], label='seg:'+str(seg))
                ax[1].plot(time_arr, q_d[:, seg, strain], label='seg:'+str(seg))
                ax[2].plot(time_arr, q[:, seg, strain], label='seg:'+str(seg))
            plt.xlabel('Time [s]')
            plt.show()

        # STEP 8: Append to the saving variables
        X_i = np.concatenate(
            (q.reshape((T,-1)), q_d.reshape((T,-1))), 
            axis=1
        )
        Xdot_i = np.concatenate(
            (q_d.reshape((T,-1)), q_dd.reshape((T,-1))),
            axis=1
        )
        
        X.append(X_i)
        Xdot.append(Xdot_i)

        np.save(f"results/ns-{num_segments}/config_data/" + "X.npy", X)
        np.save(f"results/ns-{num_segments}/config_data/" + "Xdot.npy", Xdot)
