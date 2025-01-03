from collections import Counter
import typer
from ..vector.classifiers.linear_svm import LinearSVMClassifier
from ..vector.classifiers.logistic_regression import LogisticRegressionClassifier
from ..core.config_manager import read_config
import os
import numpy as np

CLASSIFIERS = {
    "svm": LinearSVMClassifier,
    "logistic": LogisticRegressionClassifier
}

def initialize_classifier(vector_db):
    """Initialize and save a new classifier model."""
    config = read_config()
    classifier_type = config.get('default_classifier', 'svm')
    
    if classifier_type not in CLASSIFIERS:
        typer.secho(f"Invalid classifier type: {classifier_type}", err=True, fg=typer.colors.RED)
        return None
        
    model_path = config.get('classifier_model_path')
    if not model_path:
        model_path = os.path.expanduser(config['clustering_path'])
    else:
        model_path = os.path.expanduser(model_path)
    
    # Get all pre-calculated embeddings and their metadata from the vector database
    docs = vector_db.emails_collection.get(include=['embeddings', 'metadatas'])
    all_embeddings = np.array(docs['embeddings'])
    folders = [metadata['folder'] for metadata in docs['metadatas']]
    
    # Create and fit new classifier model
    classifier = CLASSIFIERS[classifier_type]()
    classifier.fit(all_embeddings, folders=folders)
    
    # Save the model
    classifier.save_model(model_path)
    return classifier

def get_classifier():
    """Load the existing classifier model."""
    config = read_config()
    classifier_type = config.get('default_classifier', 'svm')
    
    if classifier_type not in CLASSIFIERS:
        typer.secho(f"Invalid classifier type: {classifier_type}", err=True, fg=typer.colors.RED)
        return None
        
    model_path = config.get('classifier_model_path')
    if not model_path:
        model_path = os.path.expanduser(config['clustering_path'])
    else:
        model_path = os.path.expanduser(model_path)
        
    classifier = CLASSIFIERS[classifier_type]()
    classifier.load_model(model_path)
    return classifier

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
    # Get classifier model
    classifier = get_classifier()
    if not classifier:
        return
    
    for idx, mail in new_emails.iterrows():
        try:
            # Look up embeddings for this email's uuid in vector db
            email_ids = [f"{mail['uuid']}_{i}" for i in range(len(mail['paragraphs']))]
            email_docs = vector_db.emails_collection.get(
                ids=email_ids,
                include=['embeddings']
            )
            embeddings = email_docs['embeddings']
            
            # Use the LRClassifier to predict the folder directly
            predicted_folder = classifier.classify_email(embeddings)
            
            if predicted_folder and predicted_folder != "UNKNOWN":
                email_handler.move_mail([mail['uid']], predicted_folder)
                print(f"Moved email {mail['uuid']} to folder: {predicted_folder}")
            else:
                print(f"No valid folder prediction for email {mail['uuid']}")
                
        except Exception as e:
            print(f"Classification failed for {mail['uuid']}: {e}")
