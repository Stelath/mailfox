from typing import List, Dict
import numpy as np
from sklearn.svm import LinearSVC
from collections import Counter
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class LinearSVMClassifier:
    def __init__(self):
        self.model = LinearSVC(max_iter=1000)
        self.folder_mapping = None

    def fit(self, embeddings: np.ndarray, folders: List[str]) -> Dict:
        """
        Fit the SVM model and create folder mappings.
        
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
        
        # Split data for training and validation
        X_train, X_val, y_train, y_val = train_test_split(embeddings, y, test_size=0.2, random_state=42)
        
        # Fit the model
        self.model.fit(X_train, y_train)
        
        # Calculate metrics
        y_pred = self.model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred, average='weighted')
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'num_samples': len(folders),
            'num_folders': len(unique_folders)
        }
        
        return metrics

    def classify_email(self, email_embeddings: List[np.ndarray]) -> str:
        """
        Classify an email into a folder using SVM on the embeddings.

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

    def save_model(self, path, metrics=None):
        """Save the model, folder mapping, and metrics to disk"""
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'folder_mapping': self.folder_mapping,
                'metrics': metrics
            }, f)

    def load_model(self, path) -> Dict:
        """Load the model, folder mapping, and metrics from disk"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.folder_mapping = data['folder_mapping']
            return data.get('metrics', {})
