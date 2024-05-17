import cv2
import numpy as np

def frame_capture(path):
    # Path to video file 
    vidObj = cv2.VideoCapture(path) 
  
    # Used as counter variable 
    count = 0
  
    # checks whether frames were extracted 
    success = 1

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
    