#HW7-2-(1) Use the k-means algorithm to cluster the data into 2 clusters. Start with centroids m1 = (1; 1; 0) and
#          m2 = (2; 3; 0). Output centroids and clusters.
# ByeongKyu Park (byeonggyu.park)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# dataset
D = np.array([
    [1, 1, 0],
    [1, 0, 1],
    [2, 0, 0],
    [1, 3, 1],
    [3, 3, 2],
    [3, 0, 0],
    [0, 2, 1],
    [2, 3, 0],
    [0, 0, 0],
    [2, 1, 3]
])

# the number of clusters
k = 2

# init centroids(has not been used for this homework)
def initialize_centroids(points, k):
    # Randomly choose the first centroid
    centroids = [points[np.random.choice(len(points))]]
    for _ in range(1, k):
        # clac distances from the chosen centroids
        distances = np.sqrt(((points - np.array(centroids))**2).sum(axis=1))
        # pick a point with the max dist to all existing centroids (for ties, pick randomly)
        furthest = np.random.choice(np.where(distances == np.max(distances))[0])
        centroids.append(points[furthest])
    return np.array(centroids)

# initialize centroids
#centroids = initialize_centroids(D, k)
centroids = np.array([[1, 1, 0], [2, 3, 0]])

# assign nearest centroid
def assign_points_to_centroids(points, centroids):
    distances = np.sqrt(((points - centroids[:, np.newaxis])**2).sum(axis=2))
    # for a tie, pick a random centroid among min dists
    closest_centroid = np.array([np.random.choice(np.flatnonzero(dist == dist.min())) for dist in distances.T])
    return closest_centroid

# compute the centroids
def update_centroids(points, assignments, k):
    new_centroids = np.array([points[assignments == i].mean(axis=0) for i in range(k)])
    return new_centroids

# k-means clustering
def k_means(points, centroids, k):
    converged = False
    while not converged:
        # decide labels of each point
        closest_centroid = assign_points_to_centroids(points, centroids)
        
        # recompute the centroids
        new_centroids = update_centroids(points, closest_centroid, k)
        
        # converging ?
        converged = np.all(centroids == new_centroids)
        centroids = new_centroids
    
    return centroids, closest_centroid

# run k-means clustering
final_centroids, closest_centroids = k_means(D, centroids, k)

# prints the final centroids
print("Centroids: ")
print(final_centroids)
print()

#prints labels of each point
print("Labels: ")
cluster_labels = ['Cluster {}'.format(i) for i in closest_centroids]
print(cluster_labels)


# 3d scatter plotting with grid lines
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# data points
scatter = ax.scatter(D[:, 0], D[:, 1], D[:, 2], c=closest_centroids, s=50, cmap='viridis', depthshade=False)

# centroids
ax.scatter(final_centroids[:, 0], final_centroids[:, 1], final_centroids[:, 2], c='red', s=100, marker='x')

# grids (just for visibilities)
for i in range(D.shape[0]):
    ax.plot([D[i,0], D[i,0]], [D[i,1], D[i,1]], [0, D[i,2]], '--', linewidth=0.5, color='grey', alpha=0.7)
    ax.plot([D[i,0], D[i,0]], [0, D[i,1]], [D[i,2], D[i,2]], '--', linewidth=0.5, color='grey', alpha=0.7)
    ax.plot([0, D[i,0]], [D[i,1], D[i,1]], [D[i,2], D[i,2]], '--', linewidth=0.5, color='grey', alpha=0.7)

# labels
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('X3')

# title
ax.set_title('2-Means Clustering (with Grid Lines)')

# plotting
plt.show()