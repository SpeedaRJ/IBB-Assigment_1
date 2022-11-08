# Library imports used for implementing the LBP descriptor
import os
import numpy as np
import cv2
import copy
from scipy.spatial.distance import cdist
from scipy.ndimage import generic_filter
# Import for baseline testing with skimage
from skimage import feature
# Import for saving data
import pandas

# Defining global variables with different parameter values
PATH = "./awe"
IMAGE_SIZE = [(64, 64), (128, 128), (256, 256)]
R = [1, 2, 3]
L = [8, 16, 32]
HIST_AREA = [16, 32, 64]
STEP = [1, 2, 4]


def accuracy_score(array, metric="euclidean"):  # Accuracy score function
    # Computing the distance matrix using cdist from scipy with specified metric
    distance_matrix = cdist(
        array, array, metric=metric)
    # Filling up the diagonal with infinity to avoid 100% recognition
    np.fill_diagonal(distance_matrix, np.inf)
    # Computing the truth vector or correctly predicted classes
    truth_vector = np.array(
        [i // 10 == pred // 10 for i, pred in enumerate(np.argmin(distance_matrix, 1))])
    # Returning the ratio of correctly predicted classes to all processed images
    return np.sum(truth_vector) / truth_vector.shape[0]


# Function to create a histogram for every image
def make_histograms(array, area, image_size, bins=2**L):
    for index, image in enumerate(array):
        # Calling the histogram making function for a single image
        array[index] = make_hist_for_image(image, area, image_size, bins)
    return array


# Function to create a concatenated histogram for one image
def make_hist_for_image(image, area, image_size, bins):
    histogram = []  # empty array to which to add area histograms
    # Looping through the rows and columns based on area size
    for row in range(0, image_size[0], area):
        for col in range(0, image_size[1], area):
            # Using numpy implemented histogram function for an area
            hist, _ = np.histogram(
                image[row:row+area, col:col+area].astype(int), bins=np.arange(bins + 1))
            # Adding area histogram to the histogram for the whole image
            histogram.extend(hist)
    return np.array(histogram)


def make_lbp_window(r, l):  # Function to create the window for processing LBP area
    # Computing the rotational indices of applicable pixels
    indices = [np.around(np.array(
        (r * np.sin(2*u*np.pi/l), r * np.cos(2*u*np.pi/l)))).astype(int) for u in range(l)]
    # Creating a empty window of values
    window = np.zeros((1 + np.ptp(indices), 1 + np.ptp(indices)))
    # Finding the center node
    window_center = (window.shape[0] // 2, window.shape[1] // 2)
    window[window_center] = 1
    # Setting other nodes based on center node coordinates and computed indices
    for x in indices:
        cords = x + window_center
        window[cords[0], cords[1]] = 1
    return window


def simple_lbp(values):  # Function to compute the simple LBP
    # Taking and removing the center value so it doesn't process
    center = np.take(values, values.size // 2)
    values = np.delete(values, values.size // 2)
    # Calculating the differences/binary values of neighboring pixels
    values = np.array([(value - center >= 0) * 2 **
                      index for index, value in enumerate(values)])
    return np.sum(values)  # Returning their sum as LBP feature


# Function to compute the bitwise changes
def uniform_calculation(values, center_value):
    # Computing the first part of the equation
    u_1 = np.abs(int((values[-1] - center_value) >= 0) -
                 int((values[0] - center_value) >= 0))
    # Computing the second part of the equation
    u_2 = np.sum(np.abs((values[1:] - center_value) >= 0).astype(int) -
                 ((values[:-1] - center_value) >= 0).astype(int))
    # Returning the required truth value
    return (u_1 + u_2) <= 2


def uniform_lbp(values):  # Function to compute uniform LBP
    # Taking and removing the center value so it doesn't process
    center = np.take(values, values.size // 2)
    values = np.delete(values, values.size // 2)
    # Returning the correct value of the pixel based on uniform_calculation truth value
    if uniform_calculation(values, center):
        return np.sum((values - center) >= 0)
    return L + 1
