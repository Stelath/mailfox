import chromadb
from chromadb.utils import embedding_functions

import numpy as np

class VectorDatabase():
    def __init__(self, db_path="./chroma_db/"):
        self.chroma_client = chromadb.PersistentClient(db_path)
        
        self.default_ef = embedding_functions.DefaultEmbeddingFunction()
        self.emails_collection = self.chroma_client.get_or_create_collection(name="emails", embedding_function=self.default_ef)
    
    def is_emails_empty(self):
        docs = self.emails_collection.get(include=[])
        return len(docs['ids']) == 0
    
    def embed(self, text: list[str]):
        return self.default_ef(text)
    
    def embed_email(self, email: dict):
        embeddings = np.array(self.default_ef([email['from'], email['subject'], email['body']]))
        embedding = embeddings[0] * 0.3 + embeddings[1] * 0.2 + embeddings[2] * 0.5
        embedding = embedding.reshape(1, -1)
        
        return embedding
    
    def get_all_embeddings(self, collection=None):
        docs = self.emails_collection.get(include=['embeddings'])
        ids = docs['ids']
        embeddings = np.array(docs['embeddings'])
        
        return {'ids': ids, 'embeddings': embeddings}

