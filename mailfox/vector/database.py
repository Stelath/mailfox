import chromadb
import numpy as np


class VectorDatabase():
    def __init__(self, db_path="./chroma_db/"):
        self.chroma_client = chromadb.PersistentClient(db_path)
        self.emails_collection = self.chroma_client.create_collection(name="emails")
    
    def get_all_vectors(self, collection=None):
        docs = collection.get(include=['embeddings'])
        ids = [doc.id for doc in docs]
        vectors = np.array([v.embeddings for v in docs])
        
        return {'ids': , 'vectors': vectors}

