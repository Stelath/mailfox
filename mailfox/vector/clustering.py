import pickle

from sklearn.cluster import HBDSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

class Clustering():
    def __init__(self, data, min_cluster_size=5):
        self.data = data
        self.clusterer = HBDSCAN(min_cluster_size=min_cluster_size)
        self.pca = PCA(n_components=0.95, svd_solver='auto')
        
        self.data = self.pca.fit_transform(self.data)
    
    def fit(self):
        self.hbdscan.fit(self.data)
        
    def approximate_predict(self, vector, *, threshold=0.5):
        vector = self.pca.transform(vector)
        
        nbrs = NearestNeighbors(n_neighbors=1).fit(self.clusterer.centroids_)
        distances, indices = nbrs.kneighbors(vector.reshape(1, -1))

        if distances[0][0] < threshold:
            return self.clusterer.labels_[indices[0][0]]
        else:
            return -1  # Outlier
    
    def save_model(self, path):
        pickle.dump((self.clusterer, self.pca), open(path, 'wb'))
    
    def load_model(self, path):
        self.clusterer, self.pca = pickle.load(open(path, 'rb'))