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
