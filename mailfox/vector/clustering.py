import pickle
import numpy as np

from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

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
    def __init__(self, folders, load_from_pkl = False):
        if not load_from_pkl:
            # Folders is a dictionary of folder names and their vectors
            self.centroids = {folder: np.mean(vectors, axis=0) for folder, vectors in folders.items() if len(vectors) > 0}
            self.folders = list(self.centroids.keys())
            
            self.nbrs = KNeighborsClassifier(n_neighbors=3, weights='distance').fit(list(self.centroids.values()), range(len(self.folders)))
    
    def add_folder(self, folder, vectors):
        self.folders.append(folder)
        self.centroids[folder] = np.mean(vectors, axis=0)
        self.nbrs = KNeighborsClassifier(n_neighbors=3, weights='distance').fit(list(self.centroids.values()), range(len(self.folders)))
    
    def single_predict(self, vector):
        pred = self.nbrs.predict(vector.reshape(1, -1))
        return self.folders[pred.item()]
    
    def predict(self, vectors):
        pred = self.nbrs.predict(vectors)
        return [self.folders[p] for p in pred]
            
    def save_model(self, path):
        with open(path, 'wb') as file:
            pickle.dump((self.folders, self.centroids), file)
    
    @classmethod
    def load_model(cls, path):
        try:
            with open(path, 'rb') as file:
                folders, centroids = pickle.load(file)
            
            instance = cls(None, load_from_pkl=True)
            instance.folders = folders
            instance.centroids = centroids
            instance.nbrs = KNeighborsClassifier(n_neighbors=3, weights='distance')
            instance.nbrs.fit(list(centroids.values()), range(len(folders)))
            return instance
        except Exception as e:
            print(f"An error occurred while loading the model: {e}")
            return None