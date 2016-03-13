'''
Created on Mar 13, 2016

@author: gmilani
'''

import numpy as np


def compute_stats_per_class(vector_image):
    mins = np.min(vector_image, axis = 0)
    means = np.mean(vector_image, axis = 0)
    maxs = np.max(vector_image, axis = 0)
    std = np.std(vector_image, axis = 0)
    return (mins, means, maxs, std), ('min', 'mean', 'max', 'std')