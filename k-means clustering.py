import numpy as np

# Function to compute the Euclidean distance
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# K-Means Clustering Algorithm
def kmeans(X, k, max_iters=100):
    n_samples, n_features = X.shape
    # Randomly initialize k centroids from the dataset
    centroids = X[np.random.choice(n_samples, k, replace=False)]

    # Iterate until convergence or max iterations
    for _ in range(max_iters):
        # Assign clusters
        clusters = np.array([np.argmin([euclidean_distance(x, c) for c in centroids]) for x in X])
        
        # Calculate new centroids by taking the mean of points in each cluster
        new_centroids = np.array([X[clusters == i].mean(axis=0) for i in range(k)])
        
        # If the centroids have not changed, break the loop (convergence)
        if np.all(centroids == new_centroids):
            break
        
        centroids = new_centroids

    return centroids, clusters

# Example usage
if __name__ == "__main__":
    # Create random data (e.g., 2D points)
    np.random.seed(42)
    X = np.random.rand(100, 2)
    
    # Number of clusters
    k = 3

    # Run k-means
    centroids, clusters = kmeans(X, k)
    
    print("Cluster Centers:\n", centroids)
    print("Cluster Assignments:\n", clusters)
"""
K-Means Clustering Algorithm
Introduction
K-Means is a widely used unsupervised learning algorithm designed to partition a dataset into K clusters. The goal is to group similar data points together, where each data point belongs to the cluster with the nearest mean (centroid). This algorithm is efficient, easy to understand, and commonly applied in fields like market segmentation, image compression, and pattern recognition.

Steps of the K-Means Algorithm
The K-Means algorithm follows these steps:

1. Initialize K centroids
Choose K initial centroids either randomly or by some heuristic (e.g., K-Means++). Each centroid represents a cluster.
2. Assign points to clusters
For each data point, calculate the distance (usually Euclidean distance) between the point and each centroid.
Assign each data point to the closest centroid, forming K clusters.
3. Update centroids
After assigning all points, recalculate the centroid of each cluster by taking the mean of all data points within that cluster.
The new centroid becomes the average of the points in its cluster.
4. Repeat steps 2 and 3
Continue the process of reassigning points and updating centroids until:
Convergence: The centroids no longer change, or
Max iterations: A predefined number of iterations is reached.
5. Output
The final result consists of K clusters, each with its associated data points and centroids.

Advantages of K-Means
Simplicity: Easy to implement and computationally efficient.
Scalability: Works well with large datasets.
Speed: Linear time complexity, making it suitable for large datasets.
Disadvantages of K-Means
Choice of K: The number of clusters, K, needs to be specified manually, and determining the optimal K can be challenging.
Sensitive to initialization: Poorly chosen initial centroids can lead to suboptimal clustering.
Sensitive to outliers: Outliers can significantly distort the clustering results.
Applications of K-Means
Image Compression: Reducing the number of colors in an image by clustering similar colors.
Customer Segmentation: Grouping customers based on purchasing behavior for targeted marketing.
Document Clustering: Grouping similar documents or articles based on content.
Conclusion
K-Means is a powerful yet simple clustering algorithm used extensively in data analysis and machine learning.
Despite its limitations, it remains a popular choice for its efficiency and effectiveness in grouping data into 
clusters based on similarity. The challenge lies in selecting the optimal number of clusters and handling the 
algorithm's sensitivity to initialization and outliers
"""
