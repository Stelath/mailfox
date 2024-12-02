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

    def fit(self, embeddings):
        if self.pca_components:
            self.pca = PCA(n_components=self.pca_components)
            embeddings = self.pca.fit_transform(embeddings)
        self.kmeans = KMeans(n_clusters=self.n_clusters)
        self.kmeans.fit(embeddings)

    def predict(self, embeddings):
        if self.pca:
            embeddings = self.pca.transform(embeddings)
        return self.kmeans.predict(embeddings)

    def find_closest_class(self, embeddings, classes, top_n=5):
        if self.pca:
            embeddings = self.pca.transform(embeddings)
        nbrs = NearestNeighbors(n_neighbors=top_n).fit(self.kmeans.cluster_centers_)
        distances, indices = nbrs.kneighbors(embeddings)
        predicted_classes = [classes[idx] for idx in indices.flatten()]
        return predicted_classes

    def save_model(self, path):
        with open(path, 'wb') as f:
            pickle.dump({'kmeans': self.kmeans, 'pca': self.pca}, f)

    def load_model(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.kmeans = data['kmeans']
            self.pca = data['pca']