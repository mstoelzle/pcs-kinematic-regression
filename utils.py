import cv2
import numpy as np
import matplotlib.pyplot as plt

# string of strains for plotting
string_strains = ['Bending', 'Shear', 'Axial']

def frame_capture(path):
    # Path to video file 
    vidObj = cv2.VideoCapture(path) 
    total = int(vidObj.get(cv2.CAP_PROP_FRAME_COUNT))
  
    # Used as counter variable 
    count = 0

    # vidObj object calls read 
    # function extract frames 
    success, image = vidObj.read() 

    while success:

        # Saves the frames with frame-count 
        cv2.imwrite("frame_files/frame%d.jpg" % count, image) 

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
    # add eps to the bending strain (i.e. the first column)
    theta_eps = theta + th_sign * eps

    # Compute the strains from pose through closed-form IK
    strain_data = np.zeros((T, N-1, 3))
    strain_data[:,:,0] = theta_eps / s # bending strain
    strain_data[:,:,1] = ( theta_eps / (2*s) ) * (py - (px*np.sin(theta_eps))/(np.cos(theta_eps) - 1)) # shear strain
    strain_data[:,:,2] = ( theta_eps / (2*s) ) * (-px - (py*np.sin(theta_eps))/(np.cos(theta_eps) - 1)) # axial strain

    strain_data[:,:,2] = strain_data[:,:,2] - 1 # from strain to configuration variable (axial)

    if plot:
        dt = 1e-3
        time_arr = np.arange(0.0, T*dt, dt)
        fig, ax = plt.subplots(3,1)
        for strain in range(3):
            ax[strain].grid(True)
            ax[strain].set_ylabel(string_strains[strain])

            for seg in range(N-1):
                if seg == 0:
                    ax[strain].plot(time_arr, strain_data[:, seg, strain], color='0', label='seg:'+str(seg))
                elif seg == 1:
                    ax[strain].plot(time_arr, strain_data[:, seg, strain], color='0.8', label='seg:'+str(seg))
                else:
                    ax[strain].plot(time_arr, strain_data[:, seg, strain], label='seg:'+str(seg))
        plt.xlabel('Time [s]')
        fig.suptitle('Initial strain data')
        plt.show()

    return strain_data

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
    