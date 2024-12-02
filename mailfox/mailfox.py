import os
import yaml
from tqdm.auto import tqdm
import click
import typer
from typing_extensions import Annotated
from typing import Optional
from .email_interface import EmailHandler, EmailLLM
from .vector import VectorDatabase, FolderCluster, EmbeddingFunctions
import time
from functools import partial

from concurrent.futures import ThreadPoolExecutor

app = typer.Typer()

def get_vector_db(api_key, *, email_handler):
    config = get_config()
    
    email_db_path = os.path.expanduser(config['email_db_path'])
    if os.path.exists(email_db_path) and not VectorDatabase(email_db_path).is_emails_empty():
        vector_db = VectorDatabase(email_db_path, embedding_function=config["default_embedding_function"], openai_api_key=api_key)
    elif email_handler is not None:
        typer.echo("It looks like you don't have an email database yet. Let's create one!")
        should_download_emails = typer.confirm("Would you like to download all your emails now?")
        if should_download_emails:
            vector_db = VectorDatabase(email_db_path, embedding_function=config["default_embedding_function"], openai_api_key=api_key)
            download_emails(config, email_handler, vector_db)
        else:
            typer.echo("Not downloading emails, exiting.")
            return
    else:
        typer.echo("It looks like you don't have an email database yet. Please download emails first.")

    return vector_db

@app.command()
def run():
    try:
        username, password, api_key = retrieve_credentials()
    except FileNotFoundError:
        typer.secho('Please save your credentials with the "credentials set" command!', err=True, fg=typer.colors.RED)
        return
    
    try:
        config = get_config()
    except FileNotFoundError:
        typer.secho('Please save your config with the "config set" command!', err=True, fg=typer.colors.RED)
        return

    email_handler = EmailHandler(username, password)
    vector_db = get_vector_db(api_key=api_key, email_handler=email_handler)
    if vector_db is None:
        return
    
    if config['default_classifier'] == "clustering":
        clustering_path = os.path.expanduser(config['clustering_path'])
        if os.path.exists(clustering_path):
            clustering = FolderCluster.load_model(clustering_path)
            
        else:
            folder_vectors = {}
            for folder in email_handler.get_subfolders(config['flagged_folders']):
                folder_vectors[folder] = vector_db.emails_collection.get(where={'folder': folder}, include=['embeddings'])['embeddings']
            
            clustering = FolderCluster(folder_vectors)
            clustering.save_model(clustering_path)
    elif config['default_classifier'] == "llm":
        emailLLM = EmailLLM(api_key)
    else:
        typer.secho(f"Invalid default classifier: {config['default_classifier']}", err=True, fg=typer.colors.RED)
        return
    
    while True:
        new_emails = email_handler.get_mail(filter='unseen', return_dataframe=True)
        # vector_db.store_emails(new_emails.to_dict(orient='records'))
        
        if new_emails.shape[0] == 0:
            typer.echo("No new emails found. Sleeping for 5 minutes.")
        else:
            if config['default_classifier'] == "clustering":
                for idx, mail in tqdm(new_emails.iterrows(), desc="Classifying Emails", total=new_emails.shape[0]):
                    try:
                        folder = clustering.single_predict(vector_db.embed_email(mail.to_dict()))
                        if folder is not None:
                            email_handler.move_mail([mail['uid']], folder)
                        else:
                            continue
                    except Exception as e:
                        typer.secho(f"Clustering failed for {mail['uuid']}: {e}", err=True, fg=typer.colors.RED)
            elif config['default_classifier'] == "llm":
                for idx, mail in tqdm(new_emails.iterrows(), desc="Classifying Emails", total=new_emails.shape[0]):
                    try:
                        if len(mail['body']) > (16000 * (750/1000)):
                            typer.secho(f"Email too long to send to LLM, SKIPPING", err=True, fg=typer.colors.RED)
                            continue
                        folder = emailLLM.predict_folder(mail, email_handler.get_subfolders(config['flagged_folders']))
                        email_handler.move_mail([mail['uid']], folder)
                    except ValueError as e:
                        typer.secho(f"LLM Predicted Invalid Folder, SKIPPING", err=True, fg=typer.colors.RED)
                        continue
            typer.echo("All emails classified. Sleeping for 5 minutes.")
            
        time.sleep(300)  # Wait for 5 minutes

def download_emails(config, email_handler, vector_db):
    typer.echo("Fetching all Mail")
    all_mail = email_handler.get_mail(filter='all', folders=email_handler.get_subfolders(config['flagged_folders']), return_dataframe=True)
        
    vector_db.store_emails(all_mail.to_dict(orient='records'))


# SETUP WIZARD
@app.command()
def init():
    typer.echo("Welcome to the setup wizard!")
    
    # Setting credentials
    typer.echo("Step 1: Setting your credentials")
    username = typer.prompt("Enter your email address")
    password = typer.prompt("Enter your email password", hide_input=True)
    api_key = typer.prompt("Enter your API key (optional)", default="", hide_input=True)
    set_credentials(username, password, api_key)
    
    # Configuring paths
    typer.echo("Step 2: Configuring paths")
    email_db_path = typer.prompt("Enter the path for email database", default="~/.mailfox/data/email_db")
    clustering_path = typer.prompt("Enter the path for clustering data", default="~/.mailfox/data/clustering.pkl")
    flagged_folders = typer.prompt("Enter any flagged folders (comma-separated)", default=None)
    default_classifier = typer.prompt("Enter the default classifier", default="clustering")
    default_embedding_function_choices = click.Choice([func.value for func in EmbeddingFunctions])
    default_embedding_function =  typer.prompt("Enter the default embedding function", default="st", show_choices=True, type=default_embedding_function_choices)
    set_config(email_db_path, clustering_path, flagged_folders, default_classifier, default_embedding_function)
    
    typer.echo("Step 3: Downloading emails")
    should_download_emails = typer.confirm("Would you like to download your emails to the VectorDB now?")
    if should_download_emails:
        config = get_config()
        email_handler = EmailHandler(username, password)
        vector_db = VectorDatabase(os.path.expanduser(config['email_db_path']), embedding_function=default_embedding_function, openai_api_key=api_key)
        download_emails(config, email_handler, vector_db)
    else:
        typer.echo("Email download skipped.")
    
    typer.echo("Setup complete!")


# Database Commands
database_app = typer.Typer()
app.add_typer(database_app, name="database")

@database_app.command("remove")
def remove_database():
    remove_db_flag = typer.confirm("Are you sure you want to remove the database? This action cannot be undone.", default=False)
    
    if remove_db_flag:
        if not os.path.exists(os.path.expanduser(get_config()['email_db_path'])):
            typer.echo("Database does not exist")
            return
        
        import shutil
        shutil.rmtree(os.path.expanduser(get_config()['email_db_path']))
        typer.echo("Database removed")
    else:
        typer.echo("Database removal cancelled")
        

# Clustering Commands
clustering_app = typer.Typer()
app.add_typer(clustering_app, name="clustering")

@clustering_app.command("train")
def train_clustering(re_download: Optional[bool] = False):
    config = get_config()
    username, password, api_key = retrieve_credentials()
    email_handler = EmailHandler(username, password)
    vector_db = get_vector_db(api_key=api_key, email_handler=email_handler)
    
    if re_download:
        folder_vectors = {}
        all_mail = email_handler.get_mail(filter='all', folders=email_handler.get_subfolders(config['flagged_folders']), return_dataframe=True)
        uuids = all_mail['uuid'].to_numpy()
        
        typer.echo("Updating email folders in DB")
        
        uuid_list = list(uuids)
        folder_list = list(all_mail['folder'].to_numpy())
        
        seen = set()
        unique_uuid_list = []
        unique_folder_list = []
        
        for uuid, folder in zip(uuid_list, folder_list):
            if uuid not in seen:
                seen.add(uuid)
                unique_uuid_list.append(uuid)
                unique_folder_list.append(folder)
        
        uuid_list = unique_uuid_list
        folder_list = unique_folder_list
        
        vector_db.emails_collection.update(uuid_list, metadatas=[{'folder': folder} for folder in folder_list])
    
    # for email in tqdm(db_emails, desc="Syncing Emails with DB"):
    #     current_folder = all_mail[all_mail['uuid'] == email['id']]['folder'].values[0]
    #     if email['folder'] != current_folder:
    #         vector_db.emails_collection.update(email['id'], {'folder': current_folder})
    
    clustering_path = os.path.expanduser(config['clustering_path'])
    
    if os.path.exists(clustering_path):
        typer.echo("Old clustering model detected, removing")
        os.remove(clustering_path)
    
    folder_vectors = {}
    for folder in email_handler.get_subfolders(config['flagged_folders']):
        folder_vectors[folder] = vector_db.emails_collection.get(where={'folder': folder}, include=['embeddings'])['embeddings']
    
    clustering = FolderCluster(folder_vectors)
    clustering.save_model(clustering_path)
    typer.echo("Clustering model trained and saved")

@clustering_app.command("remove")
def remove_clustering():
    remove_clustering_flag = typer.confirm("Are you sure you want to remove the clustering data? This action cannot be undone.", default=False)
    
    if remove_clustering_flag:
        if not os.path.exists(os.path.expanduser(get_config()['clustering_path'])):
            typer.echo("Clustering data does not exist")
            return
        
        os.remove(os.path.expanduser(get_config()['clustering_path']))
        typer.echo("Clustering data removed")
    else:
        typer.echo("Clustering data removal cancelled")


# CONFIG SETTINGS
config_app = typer.Typer()
app.add_typer(config_app, name="config")

@config_app.command("set")
def set_config(email_db_path: Optional[str] = None, clustering_path: Optional[str] = None, flagged_folders: Optional[str] = None, default_classifier: Optional[str] = None, default_embedding_function: Optional[str] = None):
    """
    Set configuration for the application.

    Args:
        email_db_path (Optional[str]): Path to the email database.
        clustering_path (Optional[str]): Path to the clustering data.
        flagged_folders (Optional[str]): Comma-separated list of flagged folders.
        default_classifier (Optional[str]): Default classifier to use (either clustering or llm).
        default_embedding_function (Otional[str]): Default embedding function to use (either openai or sentence-transformers (st)).
    """
    try:
        config_file = os.path.expanduser("~/.mailfox/config.yaml")
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        
        # Load existing config if the file exists
        if os.path.exists(config_file):
            with open(config_file, "r") as file:
                existing_config = yaml.safe_load(file) or {}
        else:
            existing_config = {}

        # Update config values only if they are provided
        yaml_dict = {
            "email_db_path": email_db_path if email_db_path is not None else existing_config.get("email_db_path"),
            "clustering_path": clustering_path if clustering_path is not None else existing_config.get("clustering_path"),
            "flagged_folders": [folder.strip() for folder in flagged_folders.split(",")] if flagged_folders is not None else existing_config.get("flagged_folders"),
            "default_classifier": default_classifier if default_classifier is not None else existing_config.get("default_classifier"),
            "default_embedding_function": default_embedding_function if default_embedding_function is not None else existing_config.get("default_embedding_function")
        }
        
        # Save updated config
        with open(config_file, "w") as file:
            yaml.dump(yaml_dict, file)
        
        typer.echo("Configuration saved")
    except Exception as e:
        typer.secho(f"Error saving config: {e}", err=True, fg=typer.colors.RED)

@config_app.command("list")
def list_config():
    try:
        config = get_config()
        typer.echo(yaml.dump(config))
    except FileNotFoundError:
        typer.secho("No config file found", err=True, fg=typer.colors.RED)

def get_config():
    config_file = os.path.expanduser("~/.mailfox/config.yaml")
    
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found at {config_file}")
    
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config

# CREDENTIALS
credentials_app = typer.Typer()
app.add_typer(credentials_app, name="credentials")

@credentials_app.command("set")
def set_credentials(username: Annotated[str, typer.Argument()], password: Annotated[str, typer.Argument()], api_key: Optional[str] = None):
    try:
        credentials_file = os.path.expanduser("~/.mailfox/credentials")
        os.makedirs(os.path.dirname(credentials_file), exist_ok=True)
        with open(credentials_file, "w") as f:
            f.write(f"username={username}\n")
            f.write(f"password={password}\n")
            if api_key:
                f.write(f"api_key={api_key}\n")
        typer.echo("Credentials saved")
    except Exception as e:
        typer.secho(f"Error saving credentials: {e}", err=True, fg=typer.colors.RED)

@credentials_app.command("remove")
def remove_credentials():
    try:
        credentials_file = os.path.expanduser("~/.mailfox/credentials")
        if os.path.exists(credentials_file):
            os.remove(credentials_file)
            typer.echo("Credentials removed")
        else:
            typer.echo("No credentials file found")
    except Exception as e:
        typer.secho(f"Error removing credentials: {e}", err=True, fg=typer.colors.RED)
        
def retrieve_credentials():
    credentials_file = os.path.expanduser("~/.mailfox/credentials")
    
    if not os.path.exists(credentials_file):
        raise FileNotFoundError(f"Credentials file not found at {credentials_file}")
    
    with open(credentials_file, "r") as f:
        lines = f.readlines()
        username = lines[0].split("=")[1].strip()
        password = lines[1].split("=")[1].strip()
        api_key = lines[2].split("=")[1].strip() if len(lines) > 2 else None
    return username, password, api_key


def main():
    app()
