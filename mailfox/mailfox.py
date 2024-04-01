import os

from email_interface import EmailHandler
from vector import VectorDatabase
from vector import Clustering

def main(username, password, save_path):
    email_handler = EmailHandler(username, password)
    
    db_path = os.path.join(save_path, "chroma_db")
    save_path = os.path.join(save_path, "clustering.pkl")
    if os.path.exists(save_path):
        vector_db = VectorDatabase(db_path)
        clustering = Clustering.load_model(save_path)
    else:
        os.makedirs(db_path, exist_ok=True)
        vector_db = VectorDatabase(db_path)
        clustering = Clustering(vector_db.get_all_embeddings())
        clustering.fit()
        clustering.save_model(save_path)
    
    
