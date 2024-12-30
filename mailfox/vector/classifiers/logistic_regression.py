from typing import List
import numpy as np
from sklearn.linear_model import LogisticRegression
from collections import Counter
import pickle

class LogisticRegressionClassifier:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)
        self.folder_mapping = None

    def fit(self, embeddings: np.ndarray, folders: List[str]):
        """
        Fit the logistic regression model and create folder mappings.
        
        Args:
            embeddings: Input embeddings to train on
            folders: List of folder names corresponding to embedding indices
        """
        # Create numerical labels and folder mapping
        unique_folders = list(set(folders))
        self.folder_mapping = {i: folder for i, folder in enumerate(unique_folders)}
        reverse_mapping = {folder: i for i, folder in self.folder_mapping.items()}
        
        # Convert folder names to numerical labels
        y = np.array([reverse_mapping[folder] for folder in folders])
        
        # Fit the model
        self.model.fit(embeddings, y)

    def classify_email(self, email_embeddings: List[np.ndarray]) -> str:
        """
        Classify an email into a folder using logistic regression on the embeddings.

        Args:
            email_embeddings: List of embeddings from the email

        Returns:
            str: Predicted folder name
        """
        if not email_embeddings or self.model is None:
            return "UNKNOWN"

        # Get predictions for each embedding
        predictions = []
        for embedding in email_embeddings:
            if type(embedding) == list:
                embedding = np.array(embedding)
                
            # Reshape embedding to 2D array for prediction
            embedding_2d = embedding.reshape(1, -1)
            pred = self.model.predict(embedding_2d)[0]
            predictions.append(self.folder_mapping[pred])

        if not predictions:
            return "UNKNOWN"

        # Take majority vote of predictions
        most_common_folder, _ = Counter(predictions).most_common(1)[0]
        return most_common_folder

    def save_model(self, path):
        """Save the model and folder mapping to disk"""
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'folder_mapping': self.folder_mapping
            }, f)

    def load_model(self, path):
        """Load the model and folder mapping from disk"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.folder_mapping = data['folder_mapping']