import pickle
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

class KMeansCluster():
    def __init__(self, n_clusters=10):
        self.n_clusters = n_clusters
        self.kmeans = None
        self.folder_mapping = None

    def fit(self, embeddings, folders=None):
        """
        Fit the clustering model and create folder mappings.
        
        Args:
            embeddings: Input embeddings to cluster
            folders: List of folder names corresponding to embedding indices
        """
        print("Fitting model with embeddings shape:", embeddings.shape)
        
        unique_folders = list(set(folders))
        self.n_clusters = len(unique_folders)
        
        self.kmeans = KMeans(n_clusters=self.n_clusters)
        self.kmeans.fit(embeddings)
        
        # Create folder mapping based on most common folder per cluster
        cluster_labels = self.kmeans.labels_
        self.folder_mapping = {}
        
        print("\nCreating folder mappings:")
        for cluster in range(self.n_clusters):
            # Get indices for this cluster
            cluster_indices = np.where(cluster_labels == cluster)[0]
            # Get folders for these indices
            cluster_folders = [folders[i] for i in cluster_indices]
            # Assign most common folder name to this cluster
            most_common = max(set(cluster_folders), key=cluster_folders.count)
            self.folder_mapping[str(cluster)] = most_common
            print(f"Cluster {cluster} mapped to folder: {most_common}")

    def predict(self, embeddings):
        print("Predicting for embeddings shape:", embeddings.shape)
        cluster_labels = self.kmeans.predict(embeddings)
        if self.folder_mapping:
            predictions = [self.folder_mapping[str(label)] for label in cluster_labels]
            print("Predicted folders:", predictions)
            return predictions
        return cluster_labels

    def find_closest_class(self, embeddings, classes, top_n=5):
        print("\nFinding closest classes for embedding:")
        print("Input embedding shape:", embeddings.shape)
        print("Input embedding:", embeddings)
        
        nbrs = NearestNeighbors(n_neighbors=top_n).fit(self.kmeans.cluster_centers_)
        distances, indices = nbrs.kneighbors(embeddings)
        
        print("\nNearest neighbor distances:", distances)
        
        predicted_classes = []
        for i, idx_array in enumerate(indices):
            if self.folder_mapping:
                cluster_labels = [self.folder_mapping[str(classes[int(idx)])] for idx in idx_array]
            else:
                cluster_labels = [str(classes[int(idx)]) for idx in idx_array]
            
            print(f"\nTop {top_n} closest folders for embedding {i}:")
            for j, (label, dist) in enumerate(zip(cluster_labels, distances[i]), 1):
                print(f"{j}. Folder: {label}, Distance: {dist:.4f}")
            
            predicted_classes.append(cluster_labels)
            
        return predicted_classes[0] if len(predicted_classes) == 1 else predicted_classes

    def save_model(self, path):
        with open(path, 'wb') as f:
            pickle.dump({
                'kmeans': self.kmeans,
                'folder_mapping': self.folder_mapping
            }, f)

    def load_model(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.kmeans = data['kmeans']
            self.folder_mapping = data.get('folder_mapping')
