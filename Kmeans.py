#imports cell
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from copy import deepcopy
from sklearn.metrics.pairwise import rbf_kernel as rbf
from sklearn.neighbors import NearestNeighbors as nn
from scipy.sparse.linalg import eigs as sciEigs

class kmeans:
    def __init__(self,dataset):
        self.dataset = dataset

    def kmeans(data, num_clusters=3, tolerance=0.0001, max_iter=300, init_seed=None):
        iter_num = 0
        # Number of training data
        n = data.shape[0]
        # Number of features in the data
        c = data.shape[1]
        # Generate random centers, here i use standard devation
        # and mean to ensure it represents the whole data
        if (init_seed is None):
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            centroids = np.random.randn(num_clusters, c) * std + mean
        else:
            centroids = init_seed

        # to store old centers
        old_centroids = np.zeros(centroids.shape)
        # Store new centers
        new_centroids = deepcopy(centroids)
        # generate error vector
        error = np.linalg.norm(new_centroids - old_centroids)
        # create clusters array
        clusters = np.zeros(n)
        # create distaces array
        distances = np.zeros((n, num_clusters))
        # When, after an update, the estimate of that center stays the same, exit loop
        while error > tolerance and iter_num < max_iter:
            iter_num += 1
            # Measure the distance to every center
            for i in range(num_clusters):
                distances[:, i] = np.linalg.norm(data - new_centroids[i], axis=1)
            # Assign all training data to closest center
            clusters = np.argmin(distances, axis=1)
            old_centroids = deepcopy(new_centroids)
            # Calculate mean for every cluster and update the center
            for i in range(num_clusters):
                new_centroids[i] = np.mean(data[clusters == i], axis=0)
            error = np.linalg.norm(new_centroids - old_centroids)

        return new_centroids, clusters, error, iter_num

    def Distance(x, y):
        dist = np.sum(np.abs(x - y) ** 2, axis=-1) ** (1. / 2)
        return dist