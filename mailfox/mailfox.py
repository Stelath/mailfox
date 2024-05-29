import os
import yaml
from tqdm.auto import tqdm
import typer
from typing_extensions import Annotated
from typing import Optional
from .email_interface import EmailHandler, EmailLLM
from .vector import VectorDatabase, FolderCluster
import time
from functools import partial

from concurrent.futures import ThreadPoolExecutor

app = typer.Typer()

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
    
    email_db_path = os.path.expanduser(config['email_db_path'])
    if os.path.exists(email_db_path) and not VectorDatabase(email_db_path).is_emails_empty():
        vector_db = VectorDatabase(email_db_path)
    else:
        typer.echo("It looks like you don't have an email database yet. Let's create one!")
        should_download_emails = typer.confirm("Would you like to download all your emails now?")
        if should_download_emails:
            vector_db = VectorDatabase(email_db_path)
            download_emails(config, email_handler, vector_db)
        else:
            typer.echo("Not downloading emails, exiting.")
            return
    
    if config['default_classifier'] == "clustering":
        clustering_path = os.path.expanduser(config['clustering_path'])
        if os.path.exists(clustering_path):
            clustering = FolderCluster.load_model(clustering_path)
        else:
            clustering.fit()
            clustering.save_model(clustering_path)
    elif config['default_classifier'] == "llm":
        emailLLM = EmailLLM(api_key)
    else:
        typer.secho(f"Invalid default classifier: {config['default_classifier']}", err=True, fg=typer.colors.RED)
        return
    
    while True:
        new_emails = email_handler.get_mail(filter='unseen', return_dataframe=True)
        
        if new_emails.shape[0] == 0:
            typer.echo("No new emails found. Sleeping for 5 minutes.")
        else:
            if config['default_classifier'] == "clustering":
                for idx, mail in tqdm(new_emails.iterrows(), desc="Classifying Emails", total=new_emails.shape[0]):
                    folder = clustering.predict_folder(mail)
                    email_handler.move_mail([mail['uid']], folder)
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
        
    for idx, mail in tqdm(all_mail.iterrows(), desc="Generating Database", total=all_mail.shape[0]):
        embedding = vector_db.embed_email(mail)
        
        vector_db.emails_collection.add(
            ids=[mail['uuid']],
            embeddings=embedding,
            metadatas=dict(mail.drop('uuid'))
        )


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
    set_config(email_db_path, clustering_path, flagged_folders, default_classifier)
    
    typer.echo("Step 3: Downloading emails")
    should_download_emails = typer.confirm("Would you like to download your emails to the VectorDB now?")
    if should_download_emails:
        config = get_config()
        email_handler = EmailHandler(username, password)
        vector_db = VectorDatabase(config['email_db_path'])
        download_emails(config, email_handler, vector_db)
    else:
        typer.echo("Email download skipped.")
    
    typer.echo("Setup complete!")


# CONFIG SETTINGS
config_app = typer.Typer()
app.add_typer(config_app, name="config")

@config_app.command("set")
def set_config(email_db_path: Optional[str] = None, clustering_path: Optional[str] = None, flagged_folders: Optional[str] = None, default_classifier: Optional[str] = None):
    """
    Set configuration for the application.

    Args:
        email_db_path (Optional[str]): Path to the email database.
        clustering_path (Optional[str]): Path to the clustering data.
        flagged_folders (Optional[str]): Comma-separated list of flagged folders.
        default_classifier (Optional[str]): Default classifier to use (either clustering or llm).
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
            "default_classifier": default_classifier if default_classifier is not None else existing_config.get("default_classifier")
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
