from sentence_transformers import SentenceTransformer

class Encoder():
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
    
    def encode(self, text):
        return self.model.encode(text)
    
    def encode_batch(self, texts):
        return self.model.encode(texts)
    
    def save(self, path):
        self.model.save(path)
    