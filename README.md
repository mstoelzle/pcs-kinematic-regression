# Kinematic Regression

This repository implements a Kinematic Regression approach to obtain configuration space data of a soft robot directly from pixels. The method follows roughly the following sequence:
1) From a demonstration video of a soft robot, CV techniques are used to get the cartesian space pose for each of the N marked cross sections.
2) Based on a PCS parametrization and assuming an initial N segmentation, obtain the strain data for each of these segments.
3) Iteratively join adjacent segments which have similar strain behaviour
4) Average the strain data of these new "groups of segmets" ()weighted by segment length) to get updated configurations based on new segmentation.

## Installation
* Clone this repository
* Install dependencies (see below)

## How to Use
* `PCSKinematicRegression.py` is the main script. Run this file to do the Kinematic Regression.
* `soft_manipulator_curve_fitting.py` implements the function `get_task_pose` which is responsible to extract the task space pose from the video.
* `segment_merging_algorithm.py` contains the algorithm for merging the initial N segments.
* `utils.py` has auxiliary functions used across the above three files.

## Dependencies
* numpy
* cv2
* scipy
* matplotlib
