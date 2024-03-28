import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.stats import ks_2samp
from sklearn.metrics import silhouette_samples, silhouette_score
import os
import glob
from skimage import io
from sklearn.cluster import KMeans
from kneed import KneeLocator
import numpy as np

def generate_random_distribution(data, image_size):
    x_min, x_max = data['centroid_x'].min(), data['centroid_x'].max()
    y_min, y_max = data['centroid_y'].min(), data['centroid_y'].max()
    num_points = len(data)
    random_x = np.random.uniform(low=x_min, high=x_max, size=num_points)
    random_y = np.random.uniform(low=y_min, high=y_max, size=num_points)
    return np.column_stack((random_x, random_y))

def calculate_characteristics(centroids, image_size):
    distances = squareform(pdist(centroids))
    avg_distance = np.mean(distances[np.nonzero(distances)])
    distribution_evenness = np.var(distances[np.nonzero(distances)])
    plaque_density = (len(centroids) / (image_size[0] * image_size[1])) * 10000
    center_of_image = np.array([image_size[0] / 2, image_size[1] / 2])
    dist_from_center = np.mean(cdist(centroids, [center_of_image]))
    avg_dist_to_edge = _calculate_avg_distance_to_edge(centroids, image_size)
    optimal_clusters = calculate_optimal_clusters(centroids)
    labels, cluster_centroids = perform_kmeans_clustering(centroids, optimal_clusters)
    nearest_neighbor_distance = calculate_nearest_neighbor_distance(centroids)
    

    return avg_distance, distribution_evenness, plaque_density, dist_from_center, avg_dist_to_edge, nearest_neighbor_distance, optimal_clusters, silhouette_score(cluster_centroids, labels)

def calculate_nearest_neighbor_distance(centroids):
    distances = squareform(pdist(centroids))
    np.fill_diagonal(distances, np.inf)
    nearest_neighbor_dist = np.min(distances, axis=1)
    return np.mean(nearest_neighbor_dist)

def calculate_optimal_clusters(data, max_clusters=10):
        
        wcss = []  # Within-cluster sum of squares
        for i in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
            kmeans.fit(data)
            wcss.append(kmeans.inertia_)
        
        # Use the KneeLocator to find the elbow point
        kneedle = KneeLocator(range(1, max_clusters + 1), wcss, curve='convex', direction='decreasing')
        optimal_clusters = kneedle.elbow
        
        if optimal_clusters is None:
            raise ValueError("Optimal number of clusters could not be determined automatically.")
        
        print(f"Optimal number of clusters determined to be: {optimal_clusters}")
        return optimal_clusters

def perform_kmeans_clustering(data, n_clusters):
    
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
    labels = kmeans.fit_predict(data)
    centroids = kmeans.cluster_centers_
    return labels, centroids

def _calculate_avg_distance_to_edge(centroids, image_size):
    dist_to_edge = []
    for centroid in centroids:
        distances = [centroid[0], image_size[0] - centroid[0], centroid[1], image_size[1] - centroid[1]]
        dist_to_edge.append(min(distances))
    return np.mean(dist_to_edge)

def identify_region(image_name):
    if 'HI' in image_name:
        return 'HI'
    elif 'SSCTX' in image_name:
        return 'SSCTX'
    else:
        return 'Other'

def perform_ks_test(real_data, random_data):
    return ks_2samp(real_data, random_data)


def main():
    # Define your base directories
    processed_path = "/Users/katherineridley/Projects/PlaqueDist/Processed"
    masks_path = "/Users/katherineridley/Projects/PlaqueDist/Masks"
    random_dists_path = "/Users/katherineridley/Projects/PlaqueDist/RandomDists"

    # Ensure the RandomDists directory exists
    os.makedirs(random_dists_path, exist_ok=True)

    # Listing all the CSV files in processed_path
    csv_files = glob.glob(os.path.join(processed_path, "*.csv"))
    
    results = []

    for csv_file in csv_files:
        print(f"Processing {csv_file}...")
        data = pd.read_csv(csv_file)
        
        # Identifying the corresponding mask image based on the CSV filename
        base_name = os.path.basename(csv_file).split('.')[0]
        image_file = os.path.join(masks_path, f"{base_name}.tiff")
        
        if os.path.exists(image_file):
            imagestack = io.imread(image_file)
            image_dimensions = [imagestack.shape[1], imagestack.shape[2]]
            
            # Calculate characteristics for the real distribution
            real_centroids = data[['centroid_x', 'centroid_y']].values
            real_metrics = calculate_characteristics(real_centroids, image_dimensions)
            
            # Generate random distribution and calculate characteristics
            random_centroids = generate_random_distribution(data, image_dimensions)
            random_metrics = calculate_characteristics(random_centroids, image_dimensions)
            
            # Save the random distribution to a new CSV file
            random_df = pd.DataFrame(random_centroids, columns=['centroid_x', 'centroid_y'])
            random_df.to_csv(os.path.join(random_dists_path, f"{base_name}_random.csv"), index=False)
            
            # Perform KS test between real and random distributions
            ks_stat, ks_pvalue = perform_ks_test(real_metrics, random_metrics)
            
            result = {
                "filename": base_name,
                "real_metrics": real_metrics,
                "random_metrics": random_metrics,
                "ks_stat": ks_stat,
                "ks_pvalue": ks_pvalue
            }
            results.append(result)
        else:
            print(f"Mask image {image_file} not found.")
    
    # Converting results into a DataFrame and saving to CSV for further analysis
    results_df = pd.DataFrame(results)
    results_df.to_csv("aggregated_results.csv", index=False)
    print("Processing complete. Results saved to aggregated_results.csv.")

