import numpy as np
from copy import deepcopy
from gensim.corpora import Dictionary
from gensim.models import Word2Vec, TfidfModel
import math
import random


class KMeansRecommend:
    def __init__(self,data):
        self.data = data
        self.new_centroids = None
        self.clusters = None

    def k_means(self, num_clusters=3, tolerance=0.0001, max_iter=300, init_seed=None):
        data = self.data
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

        self.new_centroids = new_centroids
        self.clusters = clusters

        return new_centroids, clusters, error, iter_num

    def recommend(self, article):
        article.tokenize().remove_stop_words().lemmatize()
        n_grams = article.get_n_grams(2)

        vocab = Dictionary([n_grams])
        corpus = [vocab.doc2bow(n_grams)]  # convert corpus to BoW format
        model = TfidfModel(corpus)

        vector = 0
        for n in model[corpus[0]]:
            vector += n[1]

        distances = []

        for c in self.new_centroids:
            distance = math.fabs(np.linalg.norm(vector - c))
            distances.append(distance)

        min = np.array(distances).argmin()

        all_recommended = []
        i = 0
        for c in self.clusters:
            if c == min:
                all_recommended.append(i)
            i += 1

        recommended_article_ids = []
        for i in range(0, 3):
            random_article = random.choice(all_recommended)
            recommended_article_ids.append(random_article)

        return recommended_article_ids
