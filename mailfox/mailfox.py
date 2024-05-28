import os
from tqdm.auto import tqdm

from email_interface import EmailHandler
from email_interface import EmailLLM
from vector import VectorDatabase
from vector import Clustering

def main(username, password, save_path):
    email_handler = EmailHandler(username, password)
    emailLLM = EmailLLM()
    
    db_path = os.path.join(save_path, "chroma_db")
    clustering_path = os.path.join(save_path, "clustering.pkl")
    if os.path.exists(save_path):
        vector_db = VectorDatabase(db_path)
        clustering = Clustering.load_model(save_path)
    else:
        os.makedirs(db_path, exist_ok=True)
        vector_db = VectorDatabase(db_path)
        
        print("Fetching all Mail")
        all_mail = email_handler.get_mail(filter='all', return_dataframe=False)
        
        for mail in tqdm(all_mail, desc="Generating Database"):
            vector_db.emails_collection.add(
                ids=mail['id'],
                documents=mail['body'],
                metadatas=mail.drop(columns=['body']).drop(columns=['id'])
            )
        
        clustering = Clustering(vector_db.get_all_embeddings())
        clustering.fit()
        clustering.save_model(clustering_path)
        
        clustering.clusterer._centroids
    
