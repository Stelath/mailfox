from sentence_transformers import SentenceTransformer

class Encoder():
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
    
    def encode(self, sentences):
        return self.model.encode(sentences)
    
    def encode_email(self, email):
        return self.encode([email['Body']])[0]
    
    def encode_emails(self, emails):
        return self.encode([email['Body'] for email in emails])