import os
import numpy as np
import cv2
import copy
from scipy.spatial.distance import cdist
from scipy.ndimage import generic_filter
# Baseline imports
from skimage import feature

import pandas

PATH = "./awe"
IMAGE_SIZE = [(64, 64), (128, 128), (256, 256)]
R = [1, 2, 3]
L = [8, 16, 32]
HIST_AREA = [16, 32, 64]
STEP = [1, 2, 4]


def accuracy_score(array, metric="euclidean"):
    distance_matrix = cdist(
        array, array, metric=metric)
    np.fill_diagonal(distance_matrix, np.inf)
    truth_vector = np.array(
        [i // 10 == pred // 10 for i, pred in enumerate(np.argmin(distance_matrix, 1))])
    return np.sum(truth_vector) / truth_vector.shape[0]


def make_histograms(array, area, image_size, bins=2**L):
    for index, image in enumerate(array):
        array[index] = make_hist_for_image(image, area, image_size, bins)
    return array


def make_hist_for_image(image, area, image_size, bins):
    histogram = []
    for row in range(0, image_size[0], area):
        for col in range(0, image_size[1], area):
            hist, _ = np.histogram(
                image[row:row+area, col:col+area].astype(int), bins=np.arange(bins + 1))
            histogram.extend(hist)
    return np.array(histogram)


def make_lbp_window(r, l):
    indices = [np.around(np.array(
        (r * np.sin(2*u*np.pi/l), r * np.cos(2*u*np.pi/l)))).astype(int) for u in range(l)]

    window = np.zeros((1 + np.ptp(indices), 1 + np.ptp(indices)))
    window_center = (window.shape[0] // 2, window.shape[1] // 2)
    window[window_center] = 1
    for x in indices:
        cords = x + window_center
        window[cords[0], cords[1]] = 1
    return window


def simple_lbp(values):
    center = np.take(values, values.size // 2)
    values = np.delete(values, values.size // 2)
    values = np.array([(value - center >= 0) * 2 **
                      index for index, value in enumerate(values)])
    return np.sum(values)


def uniform_calculation(values, center_value):
    u_1 = np.abs(int((values[-1] - center_value) >= 0) -
                 int((values[0] - center_value) >= 0))
    u_2 = np.sum(np.abs((values[1:] - center_value) >= 0).astype(int) -
                 ((values[:-1] - center_value) >= 0).astype(int))
    return (u_1 + u_2) <= 2


def uniform_lbp(values):
    center = np.take(values, values.size // 2)
    values = np.delete(values, values.size // 2)
    if uniform_calculation(values, center):
        return np.sum((values - center) >= 0)
    return L + 1
