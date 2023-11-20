from skimage import io
import os
import numpy as np
import matplotlib.pyplot as plt

# assign nearest centroid
def assign_points_to_centroids(points, centroids):
    distances = np.sqrt(((points - centroids[:, np.newaxis])**2).sum(axis=2))
    # for a tie, pick a random centroid among min dists
    closest_centroid = np.array([np.random.choice(np.flatnonzero(dist == dist.min())) for dist in distances.T])
    return closest_centroid

# compute the centroids
def update_centroids(points, assignments, k):
    new_centroids = np.zeros((k, points.shape[1]))
    for i in range(k):
        assigned_points = points[assignments == i]
        if len(assigned_points) > 0:
            new_centroids[i] = assigned_points.mean(axis=0)
    # Cast the centroid values to integers
    return new_centroids.astype(int)

def initialize_centroids(points, k):
    # Randomly choose the first centroid
    centroids = [points[np.random.choice(len(points))]]
    for _ in range(1, k):
        # Calculate distances from all points to each centroid
        distances = np.sqrt(((points - centroids[-1])**2).sum(axis=1))
        for c in centroids[:-1]:  # Exclude the last added centroid
            dist_to_c = np.sqrt(((points - c)**2).sum(axis=1))
            distances = np.minimum(distances, dist_to_c)  # Keep the smallest distances

        # Pick the point with the max distance to all existing centroids (for ties, pick randomly)
        furthest = np.random.choice(np.where(distances == np.max(distances))[0])
        centroids.append(points[furthest])
    return np.array(centroids)


# k-means clustering
def k_means(points, initial_centroids, k):
    centroids = initial_centroids
    converged = False
    while not converged:
        # decide labels of each point
        closest_centroid = assign_points_to_centroids(points, centroids)
        
        # recompute the centroids
        new_centroids = update_centroids(points, closest_centroid, k)
        
        # convergence check
        converged = np.all(centroids == new_centroids)
        centroids = new_centroids
    
    return centroids, closest_centroid


# Function to read an image and convert it to a dataset D
def image_to_dataset(image_path):
    image = io.imread(image_path)
    D = image.reshape((-1, 3))  # flatten the image to create the dataset
    return D, image.shape

# Recolor the image with the centroids
def recolor_image(D, centroids, closest_centroids):
    centroids = centroids.astype(int)   # Ensure centroids are in the correct range by casting to integers
    recolored_D = np.array([centroids[i] for i in closest_centroids])
    return recolored_D

def k_means_image(image_path, k_values, n_runs, output_dir):
    D, original_shape = image_to_dataset(image_path)
    
    # Create the output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for k in k_values:
        print(f"Processing k={k}")
        for run in range(n_runs):
            print(f"Run {run + 1}")
            initial_centroids = initialize_centroids(D, k)
            centroids, closest_centroids = k_means(D, initial_centroids, k)

            # Write results to a text file
            output_txt_filename = f'k{k}_run{run+1}_results.txt'
            output_txt_path = os.path.join(output_text_path, output_txt_filename)
            
            with open(output_txt_path, 'w') as file:
                file.write(f"Centroids for k={k}, Run {run + 1}:\n{centroids}\n")
                # Print first few assignments
                sample_assignments = closest_centroids[:100] 
                file.write(f"First 100 Cluster Assignments for k={k}, Run {run + 1}:\n{sample_assignments}\n")

            recolored_D = recolor_image(D, centroids, closest_centroids)
            recolored_image = recolored_D.reshape(original_shape).astype(np.uint8)
            
            plt.imshow(recolored_image)
            plt.title(f'k={k}, Run {run + 1} Quantized Image')
            plt.axis('off')
            
            # Save the figure
            output_filename = f'quantized_k{k}_run{run+1}.png'
            output_path = os.path.join(output_dir, output_filename)
            plt.savefig(output_path, bbox_inches='tight')
            plt.close() 

image_path = 'images/original/1.jpg' 
k_values = range(3, 11)
n_runs = 5 
output_dir = 'output_images_1'  
output_text_path = 'clustering_1'
k_means_image(image_path, k_values, n_runs, output_dir)