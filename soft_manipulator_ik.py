import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from utils import inverse_kinematics, determine_merges

# number of discretized points (cross sections)
num_cs = 21
# number of segments for getting the total length
num_segments = 1
# filtering flag
filtering = True
# scaling flag
scaling = True
# original length
params = {"l": 1e-1 * np.ones((num_segments,))}
# original distance between points
s = np.sum(params["l"]) / num_cs
# small number to avoid singularities
eps = 1e-4
# threshold for merging segments
threshold = 0.25
# string of strains for plotting
string_strains = ['Bending', 'Shear', 'Axial']

# Load the variable with the poses of each point along the trajectory
pose_data = np.load('./data_poses/trajectory_pose.npy')
pose_data[:,:,2] = pose_data[:,:,2]*np.pi/180 # convert deg to rad

# Convert from cartesian to configuration space
strain_data = inverse_kinematics(pose_data, eps, s, plot=True)
T, N_SEG, _ = strain_data.shape

# Smooth out trajectories with Savgol filter
if filtering:
    smooth_strain_data = savgol_filter(strain_data, 14*5 + 1, polyorder=3, deriv=0, axis=0)

    dt = 1e-3
    time_arr = np.arange(0.0, T*dt, dt)
    fig, ax = plt.subplots(3,1)
    for strain in range(3):
        ax[strain].grid(True)
        ax[strain].set_ylabel(string_strains[strain])

        for seg in range(N_SEG):
            if seg == 0:
                ax[strain].plot(time_arr, smooth_strain_data[:, seg, strain], color='0', label='seg:'+str(seg))
            elif seg == 1:
                ax[strain].plot(time_arr, smooth_strain_data[:, seg, strain], color='0.8', label='seg:'+str(seg))
            else:
                ax[strain].plot(time_arr, smooth_strain_data[:, seg, strain], label='seg:'+str(seg))
    plt.xlabel('Time [s]')
    fig.suptitle('Strain data after SG filter')
    plt.show()

# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.set_xlabel('Bending')
# ax.set_ylabel('Shear')
# ax.set_zlabel('Axial')
# ax.plot3D(strains_trajectory[:,0,0], strains_trajectory[:,0,1], strains_trajectory[:,0,2])
# ax.plot3D(strains_trajectory[:,1,0], strains_trajectory[:,1,1], strains_trajectory[:,1,2])
# plt.show()

# # Scale data (Max abs scaling)
# if scaling:
#     if filtering:
#         max_strain = np.max(np.abs(np.reshape(smooth_strain_data, (-1,3))), axis=0)
#         # max_strain = np.where(max_strain <= 1, 1, max_strain)
#         scaled_strain_data = smooth_strain_data / max_strain[None, None, :]
#     else:  
#         max_strain = np.max(np.abs(np.reshape(strain_data, (-1,3))), axis=0)
#         # max_strain = np.where(max_strain <= 1, 1, max_strain)
#         scaled_strain_data = strain_data / max_strain[None, None, :]

#     dt = 1e-3
#     time_arr = np.arange(0.0, T*dt, dt)
#     fig, ax = plt.subplots(3,1)
#     for strain in range(3):
#         ax[strain].grid(True)
#         ax[strain].set_ylabel(string_strains[strain])

#         for seg in range(N_SEG):
#             ax[strain].plot(time_arr, scaled_strain_data[:, seg, strain], label='seg:'+str(seg))
#     plt.xlabel('Time [s]')
#     fig.suptitle('Strain data after Max-abs scaling')
#     plt.show()

# Custom scale data (based on the variance)
if scaling:
    if filtering:
        sdt = np.sqrt(np.var(smooth_strain_data, axis=(0, 1)))
        mean = np.mean(smooth_strain_data, axis=(0,1))
        max_strain = np.max(np.abs(np.reshape(smooth_strain_data, (-1,3))), axis=0)
        # max_strain = np.where(max_strain <= 1, 1, max_strain)
        scaled_strain_data = (smooth_strain_data - mean[None, None, :]) / sdt[None, None, :]
    else:
        variances = np.var(strain_data, axis=(0, 1))  
        max_strain = np.max(np.abs(np.reshape(strain_data, (-1,3))), axis=0)
        # max_strain = np.where(max_strain <= 1, 1, max_strain)
        scaled_strain_data = strain_data / max_strain[None, None, :]

    dt = 1e-3
    time_arr = np.arange(0.0, T*dt, dt)
    fig, ax = plt.subplots(3,1)
    for strain in range(3):
        ax[strain].grid(True)
        ax[strain].set_ylabel(string_strains[strain])

        for seg in range(N_SEG):
            ax[strain].plot(time_arr, scaled_strain_data[:, seg, strain], label='seg:'+str(seg))
    plt.xlabel('Time [s]')
    fig.suptitle('Strain data after Max-abs scaling')
    plt.show()


if scaling:
    merges, new_strain_data = determine_merges(scaled_strain_data, threshold)
else:
    if filtering:
        merges, new_strain_data = determine_merges(smooth_strain_data, threshold)
    else:
        merges, new_strain_data = determine_merges(strain_data, threshold)


_, N_SEG, _ = new_strain_data.shape
dt = 1e-3
time_arr = np.arange(0.0, T*dt, dt)
fig, ax = plt.subplots(3,1)
for strain in range(3):
    ax[strain].grid(True)
    ax[strain].set_ylabel(string_strains[strain])

    for seg in range(N_SEG):
        ax[strain].plot(time_arr, new_strain_data[:, seg, strain], label='seg:'+str(seg))
    
    ax[strain].legend(loc="upper right")
plt.xlabel('Time [s]')
fig.suptitle('Strain data after Segment Merging Algorithm')
plt.show()

# Smooth out the configurations
smooth_new_strain_data = savgol_filter(new_strain_data, 6*5 + 1, polyorder=3, deriv=0, axis=0)
dt = 1e-3
time_arr = np.arange(0.0, T*dt, dt)
fig, ax = plt.subplots(3,1)
for strain in range(3):
    ax[strain].grid(True)
    ax[strain].set_ylabel(string_strains[strain])

    for seg in range(N_SEG):
        ax[strain].plot(time_arr, smooth_new_strain_data[:, seg, strain], label='seg:'+str(seg))
plt.xlabel('Time [s]')
fig.suptitle('Strain data after Segment Merging Algorithm and SG filter')
plt.show()




