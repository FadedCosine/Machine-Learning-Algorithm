
import time
import warnings

import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice

np.random.seed(0)

# ============
# Generate datasets
# ============
n_samples = 750
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5,
                                      noise=.05)
centers = [[1, 1], [-1, -1], [1, -1]]
blobs = datasets.make_blobs(n_samples=n_samples, centers=centers, cluster_std=0.4, random_state=0)
no_structure = np.random.rand(n_samples, 2), None
# ============
# Set up cluster parameters
# ============
plot_num = 1

default_base = {'quantile': .3,
                'eps': .3,
                'damping': .9,
                'preference': -200,
                'n_neighbors': 10,
                'n_clusters': 3,
                'min_samples': 10,
                'xi': 0.05,
                'min_cluster_size': 0.1}

datasets = [
    ("noisy_circles", noisy_circles, {'damping': .77, 'preference': -240,
                     'quantile': .2, 'n_clusters': 2,
                      'xi': 0.25}),
    ("blobs",blobs, {}),
    ]

for i_dataset, (dataset_name, dataset, algo_params) in enumerate(datasets):
    # update parameters with dataset-specific values
    print("+"*30 + dataset_name + "+"*30)
    params = default_base.copy()
    params.update(algo_params)

    X, labels_true = dataset
    
    # normalize dataset for easier parameter selection
    X = StandardScaler().fit_transform(X)
    # ============
    # Create cluster objects
    # ============
    kmeans = cluster.KMeans(n_clusters=params['n_clusters'])
    dbscan = cluster.DBSCAN(eps=params['eps'], min_samples=params['min_samples'])
    gmm = mixture.GaussianMixture(
        n_components=params['n_clusters'], covariance_type='full')

    clustering_algorithms = (
        ('KMeans', kmeans),
        ('DBSCAN', dbscan),
        ('GaussianMixture', gmm)
    )

    for name, algorithm in clustering_algorithms:
        t0 = time.time()
        algorithm.fit(X)

        t1 = time.time()
        if hasattr(algorithm, 'labels_'):
            labels = algorithm.labels_.astype(int)
        else:
            labels = algorithm.predict(X)

        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)
        print("="*30 + name + "="*30)
        print('Estimated number of clusters: %d' % n_clusters_)
        print('Estimated number of noise points: %d' % n_noise_)
        print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
        print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
        print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
        print("Adjusted Rand Index: %0.3f"
            % metrics.adjusted_rand_score(labels_true, labels))
        print("Adjusted Mutual Information: %0.3f"
            % metrics.adjusted_mutual_info_score(labels_true, labels))
        print("Silhouette Coefficient: %0.3f"
            % metrics.silhouette_score(X, labels))

        # Black removed and is used for noise instead.
        unique_labels = set(labels)
        colors = [plt.cm.Spectral(each)
                for each in np.linspace(0, 1, len(unique_labels))]
        fig = plt.figure()
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]

            class_member_mask = (labels == k)
            core_samples_mask = np.zeros_like(labels, dtype=bool)
            # core_samples_mask[algorithm.core_sample_indices_] = True
            xy = X[class_member_mask & core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                    markeredgecolor='k', markersize=14)

            xy = X[class_member_mask & ~core_samples_mask]
            
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                    markeredgecolor='k', markersize=6)
            plt.title('{alg_name} in {dataset}'.format(alg_name=name, dataset=dataset_name))
            plt.savefig("{alg_name}_in_{dataset}.png".format(alg_name=name, dataset=dataset_name))
            # plt.cla()
