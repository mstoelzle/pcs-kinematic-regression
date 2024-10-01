import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from utils import determine_merges, inverse_kinematics_direct
from itertools import chain

def segment_merging_algorithm(config_data, seg_length, threshold, filtering=True, scaling=True, string_strains=['Bending', 'Shear', 'Axial']):

    # Variables to save the strain data and segment lengths over the iterations for error metrics
    config_data_iterations = [config_data]
    seg_length_iterations = [seg_length]
    merges_iterations = []
    keep_merging = True
    # Loop the algorithm until no more merging occurs
    while keep_merging:
        T, N_SEG, _ = config_data.shape

        # STEP 1: Smooth out trajectories with Savgol filter
        if filtering:
            smooth_strain_data = savgol_filter(config_data, 10*5 + 1, polyorder=3, deriv=0, axis=0)

            dt = 1e-3
            time_arr = np.arange(0.0, T*dt, dt)
            fig, ax = plt.subplots(3,1)
            for strain in range(3):
                ax[strain].grid(True)
                ax[strain].set_ylabel(string_strains[strain])

                for seg in range(N_SEG):
                    if seg == 0:
                        # ax[strain].plot(time_arr, smooth_strain_data[:, seg, strain], '.-', color='0', label='seg:'+str(seg))
                        ax[strain].plot(time_arr, smooth_strain_data[:, seg, strain], color='0', label='seg:'+str(seg))
                    elif seg == 1:
                        # ax[strain].plot(time_arr, smooth_strain_data[:, seg, strain], '.-', color='0.8', label='seg:'+str(seg))
                        ax[strain].plot(time_arr, smooth_strain_data[:, seg, strain], color='0.8', label='seg:'+str(seg))
                    else:
                        # ax[strain].plot(time_arr, smooth_strain_data[:, seg, strain], '.-', label='seg:'+str(seg))
                        ax[strain].plot(time_arr, smooth_strain_data[:, seg, strain], label='seg:'+str(seg))
            plt.xlabel('Time [s]')
            fig.suptitle('Strain data after smoothing')
            plt.show()

        # fig = plt.figure()
        # ax = plt.axes(projection='3d')
        # ax.set_xlabel('Bending')
        # ax.set_ylabel('Shear')
        # ax.set_zlabel('Axial')
        # ax.plot3D(strains_trajectory[:,0,0], strains_trajectory[:,0,1], strains_trajectory[:,0,2])
        # ax.plot3D(strains_trajectory[:,1,0], strains_trajectory[:,1,1], strains_trajectory[:,1,2])
        # plt.show()

        # STEP 2: Custom scale data (manually chosen constants)
        if scaling:
            scaling_constants = np.array([(350*(np.pi/180))/0.1, 0.3, 0.3])
            if filtering:
                scaled_strain_data = smooth_strain_data / scaling_constants[None, None, :]
            else:
                scaled_strain_data = config_data / scaling_constants[None, None, :]

            dt = 1e-3
            time_arr = np.arange(0.0, T*dt, dt)
            fig, ax = plt.subplots(3,1,sharex=True)
            for strain in range(3):
                ax[strain].grid(True)
                ax[strain].set_ylabel(string_strains[strain])

                for seg in range(N_SEG):
                    # ax[strain].plot(time_arr, scaled_strain_data[:, seg, strain], '.-', label='seg:'+str(seg))
                    ax[strain].plot(time_arr, scaled_strain_data[:, seg, strain], label='seg:'+str(seg))
            plt.xlabel('Time [s]')
            fig.suptitle('Strain data after normalization')
            plt.show()

        # STEP 3: Apply the merging algorithm based on the predefined threshold
        if scaling:
            merges = determine_merges(scaled_strain_data, threshold)
        else:
            if filtering:
                merges = determine_merges(smooth_strain_data, threshold)
            else:
                merges = determine_merges(config_data, threshold)
        
        # STEP 4: Weighted average to get new strains
        if len(merges) != N_SEG:
            print(merges)
            merges_iterations.append(merges)
            # Some segments were merged
            # Step 4: Recompute strain data by averaging out among the new segment groups
            new_strain_data = np.zeros((T, len(merges), 3))
            new_seg_length = np.zeros((len(merges),))
            for i, group in enumerate(merges):
                if len(group) == 1: 
                    # single segment
                    new_strain_data[:, i, :] = config_data[:, group[0], :]
                    new_seg_length[i] = seg_length[group[0]]
                else: 
                    # need to average out strain data across the group
                    # new_strain_data[:, i, :] = np.mean(strain_data[:, group[0]:group[-1], :], axis=1)
                    new_strain_data[:, i, :] = np.average(config_data[:, group[0]:(group[-1] + 1), :], axis=1, weights=seg_length[group[0]:(group[-1] + 1)])
                
                    new_seg_length[i] = np.sum(seg_length[group[0]:(group[-1] + 1)])
            
            config_data = new_strain_data.copy()
            config_data_iterations.append(config_data)
            seg_length = new_seg_length.copy()
            seg_length_iterations.append(seg_length)

            T, N_SEG, _ = config_data.shape
            dt = 1e-3
            time_arr = np.arange(0.0, T*dt, dt)
            fig, ax = plt.subplots(3,1,sharex=True)
            for strain in range(3):
                ax[strain].grid(True)
                ax[strain].set_ylabel(string_strains[strain])
                # if strain == 0:
                #     ax[strain].set_ylim([14,16])

                for seg in range(N_SEG):
                    # ax[strain].plot(time_arr, config_data[:, seg, strain], '.-', label='seg:'+str(seg))
                    ax[strain].plot(time_arr, config_data[:, seg, strain], label='seg:'+str(seg))
                
                if N_SEG <= 5:
                    ax[strain].legend(loc="upper right")
            plt.xlabel('Time [s]')
            fig.suptitle('Strain data after Segment Merging Algorithm: Iter. ' + str(len(config_data_iterations) - 1) + f' - {N_SEG} segments')
            plt.show()

            if config_data.shape[1] == 1:
                # only one segment, can't merge more
                keep_merging = False
        else:
            # No segments were merged
            keep_merging = False

    return config_data_iterations, seg_length_iterations, merges_iterations


def segment_merging_algorithm_no_average(config_data, seg_length, threshold, pose_data, eps, filtering=True, scaling=True, string_strains=['Bending', 'Shear', 'Axial']):

    # Variables to save the strain data and segment lengths over the iterations for error metrics
    config_data_iterations = [config_data]
    seg_length_iterations = [seg_length]
    merges_iterations = []
    merges_absolute = []
    keep_merging = True
    # Loop the algorithm until no more merging occurs
    while keep_merging:
        T, N_SEG, _ = config_data_iterations[-1].shape

        # STEP 1: Smooth out trajectories with Savgol filter
        if filtering:
            smooth_strain_data = savgol_filter(config_data_iterations[-1], 10*5 + 1, polyorder=3, deriv=0, axis=0)

            dt = 1e-3
            time_arr = np.arange(0.0, T*dt, dt)
            fig, ax = plt.subplots(3,1)
            for strain in range(3):
                ax[strain].grid(True)
                ax[strain].set_ylabel(string_strains[strain])

                for seg in range(N_SEG):
                    if seg == 0:
                        # ax[strain].plot(time_arr, smooth_strain_data[:, seg, strain], '.-', color='0', label='seg:'+str(seg))
                        ax[strain].plot(time_arr, smooth_strain_data[:, seg, strain], color='0', label='seg:'+str(seg))
                    elif seg == 1:
                        # ax[strain].plot(time_arr, smooth_strain_data[:, seg, strain], '.-', color='0.8', label='seg:'+str(seg))
                        ax[strain].plot(time_arr, smooth_strain_data[:, seg, strain], color='0.8', label='seg:'+str(seg))
                    else:
                        # ax[strain].plot(time_arr, smooth_strain_data[:, seg, strain], '.-', label='seg:'+str(seg))
                        ax[strain].plot(time_arr, smooth_strain_data[:, seg, strain], label='seg:'+str(seg))
            plt.xlabel('Time [s]')
            fig.suptitle('Strain data after smoothing')
            plt.show()

        # fig = plt.figure()
        # ax = plt.axes(projection='3d')
        # ax.set_xlabel('Bending')
        # ax.set_ylabel('Shear')
        # ax.set_zlabel('Axial')
        # ax.plot3D(strains_trajectory[:,0,0], strains_trajectory[:,0,1], strains_trajectory[:,0,2])
        # ax.plot3D(strains_trajectory[:,1,0], strains_trajectory[:,1,1], strains_trajectory[:,1,2])
        # plt.show()

        # STEP 2: Custom scale data (manually chosen constants)
        if scaling:
            scaling_constants = np.array([(350*(np.pi/180))/0.1, 0.3, 0.3])
            if filtering:
                scaled_strain_data = smooth_strain_data / scaling_constants[None, None, :]
            else:
                scaled_strain_data = config_data_iterations[-1] / scaling_constants[None, None, :]

            dt = 1e-3
            time_arr = np.arange(0.0, T*dt, dt)
            fig, ax = plt.subplots(3,1,sharex=True)
            for strain in range(3):
                ax[strain].grid(True)
                ax[strain].set_ylabel(string_strains[strain])

                for seg in range(N_SEG):
                    # ax[strain].plot(time_arr, scaled_strain_data[:, seg, strain], '.-', label='seg:'+str(seg))
                    ax[strain].plot(time_arr, scaled_strain_data[:, seg, strain], label='seg:'+str(seg))
            plt.xlabel('Time [s]')
            fig.suptitle('Strain data after normalization')
            plt.show()

        # STEP 3: Apply the merging algorithm based on the predefined threshold
        if scaling:
            merges = determine_merges(scaled_strain_data, threshold)
        else:
            if filtering:
                merges = determine_merges(smooth_strain_data, threshold)
            else:
                merges = determine_merges(config_data_iterations[-1], threshold)
        
        # STEP 4: New Inverse Kinematics
        if len(merges) != N_SEG:
            # Some segments were merged

            # Check the merging relative to the initial segmentation
            merges_iterations.append(merges)
            if merges_absolute == []:
                merges_absolute.append(merges)
            else:
                merges_absolute_i = []
                for group in merges:
                    merges_absolute_i.append(list(chain.from_iterable([merges_absolute[-1][i] for i in group])))
                merges_absolute.append(merges_absolute_i)
            
            print(merges_absolute[-1])
            # Step 4: Recompute strain data by doing new inverse kinematics
            new_strain_data = np.zeros((T, len(merges_absolute[-1]), 3))
            new_seg_length = np.zeros((len(merges_absolute[-1]),))
            for i, group in enumerate(merges_absolute[-1]):
                if len(group) == 1: 
                    # single segment
                    new_strain_data[:, i, :] = config_data[:, group[0], :]
                    new_seg_length[i] = seg_length[group[0]]
                else: 
                    # compute relative pose of the final cross-section w.r.t. to the initial cross-section of the merged segment
                    pose_end_abs = pose_data[:, 1 + group[-1], :]
                    pose_end_rel = pose_end_abs - pose_data[:, group[0], :]
                    for t in range(pose_end_abs.shape[0]):
                        # theta = pose_end_abs[t,2]
                        theta = pose_data[t, group[0], 2]
                        pose_end_rel[t,:2] = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]) @ pose_end_rel[t,:2]

                    # find the length of the merged segment
                    new_seg_length[i] = np.sum(seg_length[group[0]:(group[-1] + 1)])

                    config_data_merged_segment = inverse_kinematics_direct(pose_end_rel, eps, new_seg_length[i])
                    new_strain_data[:,i,:] = config_data_merged_segment
                    

                    # new_strain_data[:, i, :] = np.mean(strain_data[:, group[0]:group[-1], :], axis=1)
                    # new_strain_data[:, i, :] = np.average(config_data[:, group[0]:(group[-1] + 1), :], axis=1, weights=seg_length[group[0]:(group[-1] + 1)])
                
                    # new_seg_length[i] = np.sum(seg_length[group[0]:(group[-1] + 1)])
            
            # config_data = new_strain_data.copy()
            config_data_iterations.append(new_strain_data)
            # seg_length = new_seg_length.copy()
            seg_length_iterations.append(new_seg_length)

            T, N_SEG, _ = config_data_iterations[-1].shape
            dt = 1e-3
            time_arr = np.arange(0.0, T*dt, dt)
            fig, ax = plt.subplots(3,1,sharex=True)
            for strain in range(3):
                ax[strain].grid(True)
                ax[strain].set_ylabel(string_strains[strain])
                # if strain == 0:
                #     ax[strain].set_ylim([14,16])

                for seg in range(N_SEG):
                    # ax[strain].plot(time_arr, config_data[:, seg, strain], '.-', label='seg:'+str(seg))
                    ax[strain].plot(time_arr, config_data_iterations[-1][:, seg, strain], label='seg:'+str(seg))
                
                if N_SEG <= 5:
                    ax[strain].legend(loc="upper right")
            plt.xlabel('Time [s]')
            fig.suptitle('Strain data after Segment Merging Algorithm: Iter. ' + str(len(config_data_iterations) - 1) + f' - {N_SEG} segments')
            plt.show()

            if config_data_iterations[-1].shape[1] == 1:
                # only one segment, can't merge more
                keep_merging = False
        else:
            # No segments were merged
            keep_merging = False

    return config_data_iterations, seg_length_iterations, merges_iterations

