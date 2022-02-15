
# Adapted from: https://github.com/iamsiter/FastMap-algorithm-in-Python/blob/master/FastMapUpdated.ipynb

# ------------------------------------------------------------------------------
# SCRUBBING TOGETHER COMPAS DATA
# ------------------------------------------------------------------------------

from utils import *
from get_data import *

#1 Importing the libraries
import numpy as np
from numpy import mean
from numpy import std
import matplotlib.pyplot as plt
import pandas as pd
import math

import lime
import lime.lime_tabular
import shap
from copy import deepcopy

from sklearn.preprocessing import StandardScaler



#2 Importing the COMPAS dataset
# Size = 6172
# Features: criminal history, demographics, COMPAS risk score, jail and prison time
# Positive class: High Risk (81.4%)
# Sensitive Feature: African-American (51.4%)
params = Params("model_configurations/experiment_params.json")
X, y, cols = get_and_preprocess_compas_data(params)
features = [c for c in X]
race_indc = features.index('race')

X = X.values
#list of column indexes
c_cols = [features.index('c_charge_degree_F'), features.index('c_charge_degree_M'), features.index('two_year_recid'), features.index('race'), features.index("sex_Male"), features.index("sex_Female")]

#2b Feature Scaling
#compute the mean and std to be used for later scaling.
ss = StandardScaler().fit(X)
#perform standardization by centering and scaling.
X = ss.transform(X)

#3 creating the pertubations from lime manually
lime_points = []
for _ in range(3):
	p = np.random.normal(0,1,size=X.shape)

	X_p = X + p
	X_p[:, c_cols] = X[:, c_cols]

	for row in X_p:
		for c in c_cols:
			row[c] = np.random.choice(X_p[:,c])

	lime_points.append(X_p)

lime_points = np.vstack(lime_points)
# 1 is the perturbed class & 0 is the OG class
#p = pertubated points & how far they move it...
p = [1 for _ in range(len(lime_points))]
#iid = independent variables with identical distributions & put a 0 for each var in the range of length of X
iid = [0 for _ in range(len(X))]
#stacks to pertubations and the original standardized numpy X
all_x = np.vstack((lime_points,X))
#creates an array of perturbed points & points with the same probability distribution
#--> use for classifier; check to see if it's right
all_y = np.array(p + iid)

#downsizing the all_y pertubation data from 24688 to 6172
unbal_y = np.where((all_y == 0), 0, 1)
# print("Viewing the imbalanced target vector:\n", unbal_y)
class0_og = np.where(unbal_y == 0)[0]
class1_p = np.where(unbal_y == 1)[0]

class0_len = len(class0_og)
class1_len = len(class1_p)

print("class 0 length: ", class0_len)
print("class 1 length: ", class1_len)

class1_p_downsampled = np.random.choice(class1_p, size=class0_len, replace=False)
ds_all_y = np.hstack((unbal_y[class1_p_downsampled], unbal_y[class0_og]))

#downsample all_x using the indexes from ds_all_y
ds_xp = all_x[class1_p_downsampled]
ds_all_x = np.vstack((ds_xp, all_x[class0_og]))

print("-------------------------------------")
print("Data Downsized & Stacked Together")
print("-------------------------------------")

print("-------------------------------------")
print("PCA Begins")
print("-------------------------------------")
# Reduce the dimensionality of data to 2 features
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca_results = pca.fit_transform(ds_all_x)

print("PCA Results", pca_results)
print("Shape of PCA Results:", pca_results.shape[0])

# ------------------------------------------------------------------------------
# Recursive Divisive Clustering K = 2
# Adapted from: https://vitalflux.com/hierarchical-clustering-explained-with-python-example/
# and https://towardsdatascience.com/bisecting-k-means-algorithm-clustering-in-machine-learning-1bd32be71c1c
# ------------------------------------------------------------------------------

print("-------------------------------------")
print("Bisecting Kmeans: Homemade Kmeans ")
# adapted from: https://github.com/munikarmanish/kmeans/blob/master/kmeans/__init__.py
print("-------------------------------------")

def visualize_clusters(clusters):
    """
    Visualizes the first 2 dimensions of the data as a 2-D scatter plot.
    """
    plt.figure()
    if points.shape[1] < 2:
        points = np.hstack([points, np.zeros_like(points)])
    plt.plot(points[:,0], points[:,1], 'o')
    plt.show()

def SSE(points):
    """
    Calculates the sum of squared errors for the given list of data points.
    Args:
        points: array-like
            Data points
    Returns:
        sse: float
            Sum of squared errors
    """
    centroid = np.mean(points, 0)
    errors = np.linalg.norm(points-centroid, ord=2, axis=1)
    return np.sum(errors)

def kmeans(points, k, epochs, max_iter, verbose):
    """
    Clusters the list of points into `k` clusters using k-means clustering
    algorithm.
    Args:
        points: array-like
            Data points
        k: int
            Number of output clusters
        epochs: int
            Number of random starts (to find global optima)
        max_iter: int
            Max iteration per epoch
        verbose: bool
            Display progress on every iteration
    Returns:
        clusters: list with size = k
            List of clusters, where each cluster is a list of data points
    """
    assert len(points) >= k, "Number of data points can't be less than k"

    best_sse = np.inf
    for ep in range(epochs):
        # Randomly initialize k centroids
        np.random.shuffle(points)
        centroids = points[0:k, :]

        last_sse = np.inf
        for it in range(max_iter):
            # Cluster assignment
            clusters = [None] * k
            for p in points:
                index = np.argmin(np.linalg.norm(centroids-p, 2, 1))
                if clusters[index] is None:
                    clusters[index] = np.expand_dims(p, 0)
                else:
                    clusters[index] = np.vstack((clusters[index], p))

            # Centroid update
            centroids = [np.mean(c, 0) for c in clusters]

            # SSE calculation
            sse = np.sum([SSE(c) for c in clusters])
            gain = last_sse - sse
            if verbose:
                print((f'Epoch: {ep:3d}, Iter: {it:4d}, '
                       f'SSE: {sse:12.4f}, Gain: {gain:12.4f}'))

            # Check for improvement
            if sse < best_sse:
                best_clusters, best_sse = clusters, sse

            # Epoch termination condition
            if np.isclose(gain, 0, atol=0.00001):
                break
            last_sse = sse
    print("kmeans was called")
    return best_clusters

def bisecting_kmeans(points, k, epochs, max_iter, verbose):
    """
    Clusters the list of points into `k` clusters using bisecting k-means
    clustering algorithm. Internally, it uses the standard k-means with k=2 in
    each iteration.
    Args:
        points: array-like
            Data points
        k: int
            Number of output clusters
        epochs: int
            Number of random starts (to find global optima)
        max_iter: int
            Max iteration per epoch
        verbose: bool
            Display progress on every iteration
    Returns:
        clusters: list with size = k
            List of clusters, where each cluster is a list of data points
    """
    clusters = [points]
    split = 0
    while len(clusters) < k:
        print("length of clusters list: ", len(clusters))
        max_sse_i = np.argmax([SSE(c) for c in clusters])
        cluster = clusters.pop(max_sse_i)
        two_clusters = kmeans(cluster, 2, epochs=epochs, max_iter=max_iter, verbose=verbose)
        split +=1
        print("split: ", split)
        clusters.extend(two_clusters)
    return clusters

clustered_results = np.empty(shape = pca_results.shape)
K = math.sqrt(pca_results.shape[0])
clustered_results = bisecting_kmeans(pca_results, K, 10, 300, 0)

# print("-------------------------------------")
# print("Recursive Divisive Clustering K = 2")
# print("-------------------------------------")
# from sklearn.cluster import KMeans
# clustered_results = np.empty(shape = pca_results.shape)
# K = math.sqrt(pca_results.shape[0])
# print("K:", K )
# current_clusters = 1
# split = 0
#
# while current_clusters <= K:
#     kmeans = KMeans(n_clusters=2).fit(pca_results)
#     current_clusters += 1
#     split += 1
#     cluster_centers = kmeans.cluster_centers_
#     sse = [0]*2
#     for point, label in zip(pca_results, kmeans.labels_):
#         sse[label] += np.square(point-cluster_centers[label]).sum()
#     chosen_cluster = np.argmax(sse, axis=0)
#     chosen_cluster_data = pca_results[kmeans.labels_ == chosen_cluster]
#     pca_results = chosen_cluster_data
#
# print("new pca_results length: ", len(pca_results))
#
# print("-------------------------------------")
# print("Clustering Visualization")
# print("-------------------------------------")
#
# plt.figure()
# plt.plot(clustered_results[:,0], clustered_results[:,1], 'o')
#
# Step size of the mesh. Decrease to increase the quality of the VQ.
# h = 0.02  # point in the mesh [x_min, x_max]x[y_min, y_max].
#
# # Plot the decision boundary. For that, we will assign a color to each
# x_min, x_max = clustered_results[:, 0].min() - 1, clustered_results[:, 0].max() + 1
# y_min, y_max = clustered_results[:, 1].min() - 1, clustered_results[:, 1].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
#
# # Obtain labels for each point in mesh. Use last trained model.
# Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
#
# # Put the result into a color plot
# Z = Z.reshape(xx.shape)
# plt.figure(2)
# plt.clf()
# plt.imshow(
#     Z,
#     interpolation="nearest",
#     extent=(xx.min(), xx.max(), yy.min(), yy.max()),
#     cmap=plt.cm.Paired,
#     aspect="auto",
#     origin="lower",
# )
#
# plt.plot(clustered_results[:, 0], clustered_results[:, 1], "k.", markersize=2)
# # Plot the centroids as a white X
# centroids = kmeans.cluster_centers_
# plt.scatter(
#     centroids[:, 0],
#     centroids[:, 1],
#     marker="x",
#     s=169,
#     linewidths=3,
#     color="w",
#     zorder=10,
# )
# plt.title(
#     "K-means clustering on the digits dataset (PCA-reduced data)\n"
#     "Centroids are marked with white cross"
# )
# plt.xlim(x_min, x_max)
# plt.ylim(y_min, y_max)
# plt.xticks(())
# plt.yticks(())
# plt.show()

print("-------------------------------------")
print("Perturbation & OG Scatter SLACK Plot")
print("-------------------------------------")
#BLUE scatter plot( x = results[items from the beginning through 2000/stop-1 , column_0], y = results[items from the beginning through 2000/stop-1 , column_1], blending value 0-1)
# plt.scatter(pca_results[:2000,0], pca_results[:2000,1], alpha=.1)
#ORANGE scatter plot( x = results[idk but items start through the rest of the array , column_0], y = results[idk , column_1], blending value 0-1)
# plt.scatter(pca_results[-int(X.shape[0]*.10):,0], pca_results[-int(X.shape[0]*.10):,1], alpha=.1)
#plt.show()
