import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# string of strains for plotting
string_strains = ['Bending', 'Shear', 'Axial']

def frame_capture(path, frames_folder):
    # Path to video file 
    vidObj = cv2.VideoCapture(path) 
  
    # Used as counter variable 
    count = 0

    # vidObj object calls read 
    # function extract frames 
    success, image = vidObj.read() 

    while success:

        # Saves the frames with frame-count 
        cv2.imwrite(frames_folder + "frame%d.jpg" % count, image) 

        # vidObj object calls read 
        # function extract frames 
        success, image = vidObj.read() 
  
        count += 1

def order_points(points, ind=0):
    points_new = [points[ind]]    # initialize a new list of points with the known first point
    new_idx_order = [ind]
    pcurr      = points_new[-1]       # initialize the current point (as the known point)

    while len(points_new) < len(points):
        d = np.linalg.norm(np.array(points)[:,:2] - pcurr[:2], axis=1)  # distances between pcurr and all other remaining points

        valid_idx_distances = np.where(d > 0)[0] # since pcurr belongs to points, we have to disconsider that (we want the closest point except the current point itself)
        possible_indices = valid_idx_distances[
            np.argpartition(d[valid_idx_distances], 2)[:2] # get the indices of the two closest points
        ]

        if len(points_new) == 1: # the first rectangle has to have positive y-value
            ind = possible_indices[np.array(points)[possible_indices,:][:,1] > 0][0]
            new_idx_order.append(ind)
            points_new.append(points[ind])
        else:
            # check if any of the possible two points are already in the new sorted list
            if (np.setdiff1d(list(possible_indices), new_idx_order, assume_unique=True) == possible_indices).all(): # if none of the two possible indices are in the new sorted list
                ind = valid_idx_distances[d[valid_idx_distances].argmin()] # index of the closest point
                new_idx_order.append(ind)
                points_new.append(points[ind])
            else: # if one of the two possible indices is already of the new sorted list
                ind = np.setdiff1d(list(possible_indices), new_idx_order, assume_unique=True)[0] # add the one that is not
                new_idx_order.append(ind)
                points_new.append(points[ind])

        pcurr  = points_new[-1]               # update the current point

    # # remove first entry of `new_idx_order` and reduce all indices by one 
    # # (because contour list does not identify the base rectangle)
    # new_idx_order[:] = [(entry - 1) for entry in new_idx_order]
    # new_idx_order.pop(0)
    return points_new, new_idx_order

def inverse_kinematics(pose_data, eps, s, plot=True):
    T, N, _ = pose_data.shape

    # Compute the relative tranformation between each rectangle
    relative_pose_data = [np.diff(pose_frame, axis=0) for pose_frame in list(pose_data)]
    relative_pose_data = np.array(relative_pose_data)

    # Rotate the relative position to the orientation of the previous frame
    for i in range(T):
        for j in range(1, N-1):
            theta = pose_data[i,j,2] # orientation of previous frame relative to base
            relative_pose_data[i,j,:2] = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]) @ relative_pose_data[i,j,:2]

    px = relative_pose_data[:,:,0]
    py = relative_pose_data[:,:,1]
    theta = relative_pose_data[:,:,2]
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

    # Compute the strains from pose through closed-form IK
    strain_data = np.zeros((T, N-1, 3))
    strain_data[:,:,0] = theta_eps / s # bending strain
    strain_data[:,:,1] = ( theta_eps / (2*s) ) * (py - (px*np.sin(theta_eps))/(np.cos(theta_eps) - 1)) # shear strain
    strain_data[:,:,2] = ( theta_eps / (2*s) ) * (-px - (py*np.sin(theta_eps))/(np.cos(theta_eps) - 1)) # axial strain

    config_data = strain_data.copy()
    config_data[:,:,2] = config_data[:,:,2] - 1 # from strain to configuration variable (axial)

    if plot:
        dt = 1e-3
        time_arr = np.arange(0.0, T*dt, dt)
        fig, ax = plt.subplots(3,1)
        for strain in range(3):
            ax[strain].grid(True)
            ax[strain].set_ylabel(string_strains[strain])

            for seg in range(N-1):
                if seg == 0:
                    ax[strain].plot(time_arr, config_data[:, seg, strain], color='0', label='seg:'+str(seg))
                elif seg == 1:
                    ax[strain].plot(time_arr, config_data[:, seg, strain], color='0.8', label='seg:'+str(seg))
                else:
                    ax[strain].plot(time_arr, config_data[:, seg, strain], label='seg:'+str(seg))
        plt.xlabel('Time [s]')
        fig.suptitle('Initial strain data')
        plt.show()

    return config_data

def forward_kinematics(config_data, s, eps, pose_previous_frame):
    T, _ = config_data.shape
    strain_data = config_data.copy()
    strain_data[:,2] = strain_data[:,2] + 1

    k_be = strain_data[:,0]
    sigma_sh = strain_data[:,1]
    sigma_ax = strain_data[:,2]
    # add small eps for numerical stability in bending
    k_be_sign = np.sign(k_be)
    # set zero sign to 1 (i.e. positive)
    k_be_sign = np.where(k_be_sign == 0, 1, k_be_sign)
    # add eps to bending
    k_be_eps = k_be + k_be_sign * eps

    # Compute the pose from strains through closed-form FK
    
    px = sigma_sh * (np.sin(k_be_eps * s))/k_be_eps + \
        sigma_ax * (np.cos(k_be_eps * s) - 1)/k_be_eps
    py = sigma_sh * (1 - np.cos(k_be_eps * s))/k_be_eps + \
        sigma_ax * (np.sin(k_be_eps * s))/k_be_eps
    theta = k_be_eps * s

    # Pose w.r.t the frame of the previous segment
    pose = np.array([px, py, theta]).T

    # Change pose to be w.r.t the base frame
    pose_base_frame = np.zeros((T,3))
    # Compute the angle w.r.t the base frame
    pose_base_frame[:,2] = pose_previous_frame[:,2] + pose[:,2]
    # Compute the position w.r.t the base frame
    rot_mat = np.transpose((np.array([
        [np.cos(pose_previous_frame[:,2]), -np.sin(pose_previous_frame[:,2])],
        [np.sin(pose_previous_frame[:,2]), np.cos(pose_previous_frame[:,2])]
    ])), (2,0,1))
    pose_base_frame[:,:2] = pose_previous_frame[:,:2] + np.einsum('BNi,Bi ->BN', rot_mat, pose[:,:2])
    
    # # Change pose to be w.r.t the base frame
    # # Compute the angles w.r.t the base frame
    # pose_base_frame = np.cumsum(pose, axis=2)
    # for i in range(T):
    #     for j in range(1, N):
    #         theta = pose_base_frame[i,j,2]
    #         # Compute the position w.r.t the base frame
    #         pose_base_frame[i,j,:2] = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]) @ pose[i,j,:2]


    return pose_base_frame

def compute_task_error(pose_data, config_data_itrs, seg_length_itrs, eps):
    # point coordinates of the points where the error will be calculated
    s_image_cum = np.cumsum(seg_length_itrs[0])

    T, N, _ = config_data_itrs[0].shape
    pose_data_iterations = []
    error_metric_iterations = []
    for itr in range(1, len(config_data_itrs)):
        # cumsum of the segment lengths
        s_itr_cum = np.cumsum(seg_length_itrs[itr])
        # add zero to the beginning of the array
        s_itr_cum_padded = np.concatenate([np.array([0.0]), s_itr_cum], axis=0)

        # pose of the segment frame to which the FK are being computed w.r.t
        # for the first segment, it's the base frame, which is always the same at every frame
        pose_previous_frame = np.zeros((T,3))
        prev_segment_idx = 0

        pose_itr = np.zeros((T,N,3))
        for id_seg, s_point in enumerate(s_image_cum):
            # determine in which segment the point is located
            # use argmax to find the last index where the condition is true
            segment_idx = (
                s_itr_cum.shape[0] - 1 - np.argmax((s_point > s_itr_cum_padded[:-1])[::-1]).astype(int)
            )
            
            if segment_idx != prev_segment_idx:
                pose_previous_frame = pose_itr[:, id_seg - 1, :]
                prev_segment_idx = segment_idx

            # point coordinate along the segment in the interval [0, l_segment]
            s_segment = s_point - s_itr_cum_padded[segment_idx]

            pose = forward_kinematics(config_data_itrs[itr][:,segment_idx,:], s_segment, eps, pose_previous_frame)
            pose_itr[:,id_seg,:] = pose
        
        pose_data_iterations.append(pose_itr)

        # Create the figure and axis
        fig, ax = plt.subplots()
        ax.set_xlim(-0.25, 0.25)
        ax.set_ylim(0, 0.25)

        # Initialize the scatter plots for A and B
        plot_pose_data, = ax.plot([], [], 'b-o', label='Original image')
        plot_pose_itr, = ax.plot([], [], 'r-o', label= str(itr) + ' merging iter. -> ' + str(len(s_itr_cum)) + ' segments')

        # Initialize the legend
        ax.legend()

        # Initialization function
        def init():
            plot_pose_data.set_data([], [])
            plot_pose_itr.set_data([], [])
            return plot_pose_data, plot_pose_itr

        # Update function
        def update(frame):
            plot_pose_data.set_data(pose_data[frame,:,0], pose_data[frame,:,1])
            plot_pose_itr.set_data(pose_itr[frame,:,0], pose_itr[frame,:,1])
            return plot_pose_data, plot_pose_itr

        # Create the animation
        ani = FuncAnimation(fig, update, frames=T, init_func=init, blit=True, repeat=True, interval=5)
        plt.show()
        # ani.save(filename = f"results/error_metrics/position_video_animation.gif", writer='pillow')

        print('Iteration 1:')
        error_position = np.mean(np.linalg.norm(pose_data[:,1:,:2] - pose_itr[:,:,:2], axis=2))
        print('\tmean position error: ' + str(error_position) + ' [m]')
        error_angle = np.mean(np.abs(pose_data[:,1:,2] - pose_itr[:,:,2]))*180/np.pi
        print('\tmean angle error: ' + str(error_angle) + ' [deg]')
        error_metric_iterations.append(np.array([error_position, error_angle]))

    return pose_data_iterations, error_metric_iterations

def average_lockstep_euclidean_distance(segment1, segment2):
    # return np.mean(np.sqrt(np.sum((segment1 - segment2) ** 2, axis=1)))
    return np.mean(
        np.linalg.norm(segment1 - segment2, axis=1)
    )

def determine_merges(strain_data, threshold):
    
    T, N, _ = strain_data.shape
    
    # Step 1: Compute pairwise distances
    pairwise_distances = np.zeros(N - 1)
    for i in range(N - 1):
        pairwise_distances[i] = average_lockstep_euclidean_distance(strain_data[:, i, :], strain_data[:, i + 1, :])

    fig, ax = plt.subplots(1,1)
    ax.plot(np.arange(1, len(pairwise_distances)+1), pairwise_distances, 'b:o')
    ax.grid(True)
    ax.set_ylabel('Lock-step Strain distance')
    ax.set_xlabel('Id Segment pair')
    ax.set_xticks(np.arange(1, len(pairwise_distances)+1))
    plt.show()

    # Step 2: Initial merges
    merge_candidates = []
    current_merge = [0]
    for i in range(N-1):
        if pairwise_distances[i] <= threshold:
            current_merge.append(i + 1)
        else:
            # If above the threshold, close the current group and start a new one
            merge_candidates.append(current_merge)
            current_merge = [i+1]
            # if current_merge:
            #     merge_candidates.append(list(set(current_merge)))
            #     current_merge = []

    # Add remaining group to merge_candidates
    if current_merge:
        merge_candidates.append(current_merge)

    # # Step 3: Check extremes of merging groups
    # final_merges = []
    # for group in merge_candidates:
    #     if not group:
    #         continue
    #     valid_merge = True
    #     if average_lockstep_euclidean_distance(strain_data[:, group[0], :], strain_data[:, group[-1], :]) > threshold:
    #         valid_merge = False
        
    #     if valid_merge:
    #         final_merges.append(group)
    #     else:
    #         # If the extremes are not valid, split the group into smaller valid groups
    #         subgroup = [group[0]]
    #         for i in range(1, len(group)):
    #             if average_lockstep_euclidean_distance(strain_data[:, subgroup[0], :], strain_data[:, group[i], :]) <= threshold:
    #                 subgroup.append(group[i])
    #             else:
    #                 final_merges.append(subgroup)
    #                 subgroup = [group[i]]
    #         if subgroup:
    #             final_merges.append(subgroup)

    return merge_candidates
    