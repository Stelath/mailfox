import chromadb
from chromadb.utils import embedding_functions

import numpy as np

class VectorDatabase():
    def __init__(self, db_path="./chroma_db/"):
        self.chroma_client = chromadb.PersistentClient(db_path)
        
        self.default_ef = embedding_functions.DefaultEmbeddingFunction()
        self.emails_collection = self.chroma_client.get_or_create_collection(name="emails", embedding_function=self.default_ef)
    
    def embed(self, text):
        return self.default_ef(text)
    
    def get_all_embeddings(self, collection=None):
        docs = collection.get(include=['embeddings'])
        ids = docs['ids']
        embeddings = np.array(docs['embeddings'])
        
        return {'ids': ids, 'embeddings': embeddings}

