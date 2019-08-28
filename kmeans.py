import numpy as np


def compute_new_centroids(X, cluster_assignment, k):
    new_centroids = list()
    for i in range(k):
        # Select all data points that belong to cluster ice
        member_points = np.zeros([3,1])
        for j in range(len(X)):
            if cluster_assignment[j] == i:
                member_points = np.array([X[j]])
        # Compute mean
        centroid = member_points.mean(axis=0)
        new_centroids.append(centroid)

    return np.array(new_centroids)


def k_means(k, vectors, centroids, distance):
    # Initializing clusters as np array of zeros
    cluster_assignment = np.zeros(len(vectors))
    # Error func. - Distance between new centroids and old centroids
    while True:

        error = None

        # Assigning each value to its closest cluster
        for i in range(len(vectors)):

            print(vectors[i])
            print(centroids)

            error = distance([vectors[i]], centroids)

            cluster = np.argmin(error)
            cluster_assignment[i] = cluster
        # Storing the old centroid values
        old_centroids = deepcopy(centroids)
        # Finding the new centroids by taking the average value
        centroids = compute_new_centroids(vectors, cluster_assignment, k)

        error = distance(centroids, old_centroids)

        # break loop if 0 error found
        if not error.all():
            break

    return cluster_assignment
