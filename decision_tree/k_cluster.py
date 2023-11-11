import numpy as np
import matplotlib.pyplot as plt

# Given dataset D
D = np.array([[-1, 1], [-1, 2], [0, 1], [1, 1], [2, 2], [2, 4]])

# Number of clusters
k = 2

# Function to initialize centroids
def initialize_centroids(points, k):
    # Randomly choose the first centroid
    centroids = [points[np.random.choice(len(points))]]
    for _ in range(1, k):
        # Calculate distances from the already chosen centroids
        distances = np.sqrt(((points - np.array(centroids))**2).sum(axis=1))
        # Find the data point with the maximum distance to all existing centroids
        # For ties, np.random.choice will pick one index randomly
        furthest = np.random.choice(np.where(distances == np.max(distances))[0])
        centroids.append(points[furthest])
    return np.array(centroids)

# Initialize centroids
centroids = initialize_centroids(D, k)
centroids = np.array([[-1,1],[1,1]])

# Function to assign points to the nearest centroid
def assign_points_to_centroids(points, centroids):
    distances = np.sqrt(((points - centroids[:, np.newaxis])**2).sum(axis=2))
    # If there's a tie, pick a random centroid from the ones with minimum distance
    closest_centroid = np.array([np.random.choice(np.flatnonzero(dist == dist.min())) for dist in distances.T])
    return closest_centroid

# Function to update centroids
def update_centroids(points, assignments, k):
    new_centroids = np.array([points[assignments == i].mean(axis=0) for i in range(k)])
    return new_centroids

# Function to perform k-means clustering
def k_means(points, centroids, k):
    converged = False
    while not converged:
        # Step 2: Assign points to the nearest centroid
        closest_centroid = assign_points_to_centroids(points, centroids)
        
        # Step 3: Recompute the centroids
        new_centroids = update_centroids(points, closest_centroid, k)
        
        # Check for convergence
        converged = np.all(centroids == new_centroids)
        centroids = new_centroids
    
    return centroids, closest_centroid

# Run k-means clustering
final_centroids, closest_centroids = k_means(D, centroids, k)

# Plotting the clusters
plt.scatter(D[:, 0], D[:, 1], c=closest_centroids, s=50, cmap='viridis')
plt.scatter(final_centroids[:, 0], final_centroids[:, 1], c='red', s=100, marker='x')
plt.title('2-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
