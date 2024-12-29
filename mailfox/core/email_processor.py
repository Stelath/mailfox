from collections import Counter
import typer
from ..vector import KMeansCluster
from ..core.config_manager import read_config
import os
import numpy as np

def initialize_clustering(vector_db, n_clusters=10):
    """Initialize and save a new clustering model."""
    config = read_config()
    clustering_path = os.path.expanduser(config['clustering_path'])
    
    # Get all pre-calculated embeddings and their metadata from the vector database
    docs = vector_db.emails_collection.get(include=['embeddings', 'metadatas'])
    all_embeddings = np.array(docs['embeddings'])
    folders = [metadata['folder'] for metadata in docs['metadatas']]
    
    # Create and fit new clustering model
    clustering = KMeansCluster(pca_components=None)
    clustering.fit(all_embeddings, folders=folders)
    
    # Save the model
    clustering.save_model(clustering_path)
    return clustering

def get_clustering_model():
    """Load the existing clustering model."""
    config = read_config()
    clustering_path = os.path.expanduser(config['clustering_path'])
    clustering = KMeansCluster()
    clustering.load_model(clustering_path)
    return clustering

def process_new_mail(folder, email_handler, vector_db):
    """Process new emails in a folder."""
    try:
        new_emails = email_handler.get_mail(filter="unseen", folders=[folder], return_dataframe=True)
        if not new_emails.empty:
            typer.echo(f"New emails detected in {folder}. Processing...")
            vector_db.store_emails(new_emails.to_dict(orient="records"))
            classify_emails(new_emails, vector_db, email_handler)
        else:
            typer.echo(f"No new emails in {folder}.")
    except Exception as e:
        typer.secho(f"Error processing new mail in folder {folder}: {e}", err=True, fg=typer.colors.RED)

def classify_emails(new_emails, vector_db, email_handler):
    # Get clustering model
    clustering = get_clustering_model()
    
    for idx, mail in new_emails.iterrows():
        try:
            paragraphs = mail['paragraphs']
            email_embeddings = vector_db.embed_paragraphs(paragraphs)
            predicted_classes = clustering.find_closest_class(np.array(email_embeddings), clustering.kmeans.labels_)
            
            if isinstance(predicted_classes[0], list):
                class_counts = Counter([cls for classes in predicted_classes for cls in classes])
            else:
                class_counts = Counter(predicted_classes)
            
            predicted_folder = class_counts.most_common(1)[0][0]
            
            if predicted_folder:
                email_handler.move_mail([mail['uid']], predicted_folder)
                print(f"Moved email {mail['uuid']} to folder: {predicted_folder}")
            else:
                print(f"No valid folder prediction for email {mail['uuid']}")
                
        except Exception as e:
            print(f"Classification failed for {mail['uuid']}: {e}")