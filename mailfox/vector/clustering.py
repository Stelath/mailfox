import pickle
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

class KMeansCluster():
    def __init__(self, n_clusters=10, pca_components=0.95):
        self.n_clusters = n_clusters
        self.pca_components = pca_components
        self.pca = None
        self.kmeans = None
        self.folder_mapping = None

    def fit(self, embeddings, folders=None):
        """
        Fit the clustering model and create folder mappings.
        
        Args:
            embeddings: Input embeddings to cluster
            folders: List of folder names corresponding to embedding indices
        """
        unique_folders = list(set(folders))
        self.n_clusters = len(unique_folders)
        
        if self.pca_components:
            self.pca = PCA(n_components=self.pca_components) 
            embeddings = self.pca.fit_transform(embeddings)
            
        self.kmeans = KMeans(n_clusters=self.n_clusters)
        self.kmeans.fit(embeddings)
        
        # Create folder mapping based on most common folder per cluster
        cluster_labels = self.kmeans.labels_
        self.folder_mapping = {}
        
        for cluster in range(self.n_clusters):
            # Get indices for this cluster
            cluster_indices = np.where(cluster_labels == cluster)[0]
            # Get folders for these indices
            cluster_folders = [folders[i] for i in cluster_indices]
            # Assign most common folder name to this cluster
            most_common = max(set(cluster_folders), key=cluster_folders.count)
            self.folder_mapping[str(cluster)] = most_common

    def predict(self, embeddings):
        if self.pca:
            embeddings = self.pca.transform(embeddings)
        cluster_labels = self.kmeans.predict(embeddings)
        if self.folder_mapping:
            return [self.folder_mapping[str(label)] for label in cluster_labels]
        return cluster_labels

    def find_closest_class(self, embeddings, classes, top_n=5):
        if self.pca:
            embeddings = self.pca.transform(embeddings)
        
        nbrs = NearestNeighbors(n_neighbors=top_n).fit(self.kmeans.cluster_centers_)
        distances, indices = nbrs.kneighbors(embeddings)
        
        predicted_classes = []
        for idx_array in indices:
            if self.folder_mapping:
                cluster_labels = [self.folder_mapping[str(classes[int(idx)])] for idx in idx_array]
            else:
                cluster_labels = [str(classes[int(idx)]) for idx in idx_array]
            predicted_classes.append(cluster_labels)
            
        return predicted_classes[0] if len(predicted_classes) == 1 else predicted_classes

    def save_model(self, path):
        with open(path, 'wb') as f:
            pickle.dump({
                'kmeans': self.kmeans, 
                'pca': self.pca,
                'folder_mapping': self.folder_mapping
            }, f)

    def load_model(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.kmeans = data['kmeans']
            self.pca = data['pca']
            self.folder_mapping = data.get('folder_mapping')