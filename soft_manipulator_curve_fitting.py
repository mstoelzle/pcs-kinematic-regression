import cv2
import os
import numpy as np
from scipy.interpolate import splprep, splev
from utils import frame_capture, order_points

num_segments = 1
params = {"l": 1e-1 * np.ones((num_segments,))}
ppm = 1000 / (1.5 * np.sum(params["l"]))

# Load video and extract frames
frame_extraction = False
if frame_extraction:
    video_path = "planar_pcs_ns-1.mp4"
    frame_capture(video_path)

# Check the number of frames
num_frames = len([name for name in os.listdir('frame_files') if os.path.isfile(os.path.join('frame_files', name))])

# Initialize variable to store the pose along the N points for M frames
poses_trajectory = []
for img_id in range(num_frames):
    # Load the image
    original_image = cv2.imread('frame_files/frame%d.jpg' % img_id)
    # original_image = original_image[:(int(0.9*original_image.shape[1])), :, :]

    # Convert to grayscale
    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # Thresholding to segment the soft manipulator
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    thresh = 255 - thresh

    cv2.namedWindow('Soft Manipulator with Fitted Curve', cv2.WINDOW_NORMAL)
    cv2.imshow('Soft Manipulator with Fitted Curve', thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    contours = [c for c in contours if (cv2.contourArea(c) > 350 and cv2.contourArea(c) < 1000)]

    # Draw one contour only for visualisation purposes
    rect_contours_img = original_image.copy()
    cv2.drawContours(rect_contours_img, contours, -1, (0, 255, 0), 2)
    cv2.namedWindow('Soft Manipulator with Fitted Curve', cv2.WINDOW_NORMAL)
    cv2.imshow('Soft Manipulator with Fitted Curve', rect_contours_img)
    cv2.waitKey()

    poses_frame = []
    box_list = [] # for rendering
    rect_list = []
    # poses_frame.append(np.array([0,0,0])) # the first point is always coincident to the base frame
    for i, c in enumerate(contours):
        rect_contours_img = original_image.copy()

        # Calculate the area of each contour
        area = cv2.contourArea(c)
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        # cv2.drawContours(rect_contours_img,[box],0,(0,0,255),2)
        # cv2.namedWindow('Soft Manipulator with Fitted Curve', cv2.WINDOW_NORMAL)
        # cv2.imshow('Soft Manipulator with Fitted Curve', rect_contours_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # Ignore contours that are just the outer/inner contour of the previously identified cross section
        if len(rect_list) > 0 and (np.linalg.norm(np.array(rect[0]) - np.array(rect_list[-1][0])) < 10):
            continue

        
        # Append the pose, box and rectangle
        box_list.append(box)
        rect_list.append(rect)
        poses_frame.append(np.array([
            (rect[0][0] - original_image.shape[1]//2)/ppm, # x
            ((1-0.2)*original_image.shape[0] - rect[0][1])/ppm, # y
            rect[2]
        ]))
    
    for i, box in enumerate(box_list):
        rect_contours_img = original_image.copy()
        cv2.drawContours(rect_contours_img, [box], 0, (0,0,255), 2)
        cv2.namedWindow('Soft Manipulator with Fitted Curve', cv2.WINDOW_NORMAL)
        cv2.imshow('Soft Manipulator with Fitted Curve', rect_contours_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # find the closest point to the origin (due to pixel mismatches the position of the base is not the origin)
    d = np.linalg.norm(np.array(poses_frame)[:,:2] - np.array([0,0]), axis=1)
    new_poses_frame, new_idx_order = order_points(poses_frame, ind=np.argmin(d))

    box_list[:] = [box_list[i] for i in new_idx_order]
    rect_list[:] = [rect_list[i] for i in new_idx_order]    

    for i, box in enumerate(box_list):
        rect_contours_img = original_image.copy()
        cv2.drawContours(rect_contours_img, [box], 0, (0,0,255), 2)
        cv2.namedWindow('Soft Manipulator with Fitted Curve', cv2.WINDOW_NORMAL)
        cv2.imshow('Soft Manipulator with Fitted Curve', rect_contours_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    new_poses_frame = np.array(new_poses_frame)
    new_poses_frame[0,:2] = np.array([0,0]) # the first point should have coordinates [0,0]
    # alpha are the cv2 angles
    alpha = list(new_poses_frame[:,2])
    # theta are the converted angles used for the pose
    theta = [0.0] # the first one is always 0 (base cross section)
    width_larger = [] # auxiliary variable to check if width is larger than height on the evaluated rectangles
    right_angles = [0] # variable to keep track of the successions of right angles
    for idx in range(1, new_poses_frame.shape[0]):
        w_i = rect_list[idx][1][0] # width of current rectangle
        h_i = rect_list[idx][1][1] # height of current rectangle
        if theta[idx-1] == 0.0:
            if w_i > h_i: # if width of current rectangle is larger than height
                extreme_value_index = 0
                width_larger.append(1)
            else:
                extreme_value_index = 1
                width_larger.append(-1)
            theta.append(extreme_value_index*90 - alpha[idx])
        else:
            if alpha[idx] != 90.0 and alpha[idx] != 0.0 and np.sign(w_i-h_i) == width_larger[-1]:
                theta.append(extreme_value_index*90 - alpha[idx])
            elif alpha[idx] != 90.0 and alpha[idx] != 0.0 and np.sign(w_i-h_i) != width_larger[-1]:
                # this case means the cross section has surpassed the right angle
                width_larger.append(np.sign(w_i-h_i))
                if np.sign(theta[idx-1] - theta[idx-2]) > 0: # rotating anticlockwise
                    extreme_value_index = extreme_value_index + 1
                else: # rotating clockwise
                    extreme_value_index = extreme_value_index - 1

                theta.append(extreme_value_index*90 - alpha[idx])     
            else:
                theta.append(
                    round(theta[idx-1]/90)*90
                )

    new_poses_frame[:,2] = np.array(theta)
    poses_trajectory.append(new_poses_frame)
