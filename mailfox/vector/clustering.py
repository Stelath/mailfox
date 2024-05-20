import pickle
import numpy as np

from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

class HDBCluster():
    def __init__(self, data, min_cluster_size=5, cluster_selection_epsilon=0.5, pca=True):
        self.data = data
        self.clusterer = HDBSCAN(min_cluster_size=min_cluster_size, store_centers='centroid', cluster_selection_epsilon=cluster_selection_epsilon)
        
        if pca:
            self.pca = PCA(n_components=0.95, svd_solver='auto')
            self.data = self.pca.fit_transform(self.data)
    
    def fit(self):
        self.clusterer.fit(self.data)
        
    def predict(self, vector, *, threshold=0.5):
        if hasattr(self, 'pca'):
            vector = self.pca.transform(vector)
        
        nbrs = NearestNeighbors(n_neighbors=1).fit(self.clusterer.centroids_)
        distances, indices = nbrs.kneighbors(vector.reshape(1, -1))
        
        if distances[0][0] < threshold:
            return indices[0][0]
        else:
            return -1  # Outlier
    
    def save_model(self, path):
        pickle.dump((self.clusterer, self.pca), open(path, 'wb'))
    
    def load_model(self, path):
        self.clusterer, self.pca = pickle.load(open(path, 'rb'))
        

class FolderCluster():
    def __init__(self, folders):
        # Folders is a dictionary of folder names and their vectors
        self.folders = folders.keys()
        self.centroids = {folder: np.mean(vectors, axis=0) for folder, vectors in self.folders.items()}
        self.nbrs = NearestNeighbors(n_neighbors=3, weights='distance').fit(list(self.centroids.values()))
    
    def add_folder(self, folder, vectors):
        self.folders.append(folder)
        self.centroids[folder] = np.mean(vectors, axis=0)
        self.nbrs = NearestNeighbors(n_neighbors=3, weights='distance').fit(list(self.centroids.values()))
    
    def predict(self, vector):
        distances, indices = self.nbrs.kneighbors(vector.reshape(1, -1))
        return list(self.centroids.keys())[indices[0][0]]
    
    def save_model(self, path):
        pickle.dump((self.folders, self.centroids), open(path, 'wb'))
    
    def load_model(self, path):
        self.folders, self.centroids = pickle.load(open(path, 'rb'))
        self.nbrs = NearestNeighbors(n_neighbors=3, weights='distance').fit(list(self.centroids.values()))