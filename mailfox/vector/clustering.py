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
    def __init__(self, folders=None, load_from_pkl=False, distance_threshold=0.75):
        if not load_from_pkl and folders is not None:
            # Folders is a dictionary of folder names and their vectors
            self.centroids = {folder: np.mean(vectors, axis=0) for folder, vectors in folders.items() if len(vectors) > 0}
            self.folders = list(self.centroids.keys())
        elif load_from_pkl:
            # Load when using a pre-trained model
            self.centroids = {}
            self.folders = []
        
        self.distance_threshold = distance_threshold  # Set distance cutoff threshold
    
    def add_folder(self, folder, vectors):
        """Adds a new folder and recalculates its centroid."""
        self.folders.append(folder)
        self.centroids[folder] = np.mean(vectors, axis=0)
    
    def calculate_distance(self, vector1, vector2):
        """Calculate Euclidean distance between two vectors."""
        return np.linalg.norm(vector1 - vector2)
    
    def single_predict(self, vector):
        """Predicts the nearest folder for a single vector."""
        closest_folder, closest_distance = None, float('inf')
        
        for folder, centroid in self.centroids.items():
            distance = self.calculate_distance(vector, centroid)
            if distance < closest_distance:
                closest_folder, closest_distance = folder, distance
        
        # Return the closest folder if within threshold, otherwise None
        if closest_distance <= self.distance_threshold:
            return closest_folder
        else:
            return None
    
    def predict(self, vectors):
        """Predicts the nearest folder for a list of vectors."""
        predictions = []
        for vector in vectors:
            prediction = self.single_predict(vector)
            predictions.append(prediction)
        return predictions
            
    def save_model(self, path):
        """Save the current folder structure and centroids."""
        with open(path, 'wb') as file:
            pickle.dump((self.folders, self.centroids, self.distance_threshold), file)
    
    @classmethod
    def load_model(cls, path):
        """Load a saved model from a pickle file."""
        try:
            with open(path, 'rb') as file:
                folders, centroids, distance_threshold = pickle.load(file)
            
            instance = cls(load_from_pkl=True)
            instance.folders = folders
            instance.centroids = centroids
            instance.distance_threshold = distance_threshold
            return instance
        except Exception as e:
            print(f"An error occurred while loading the model: {e}")
            return None