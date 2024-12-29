from typing import Dict, Set, List
import os
import typer
from ..vector import VectorDatabase
from ..email_interface import EmailHandler
from .config_manager import read_config

def initialize_database(email_handler: EmailHandler, vector_db: VectorDatabase, folders: List[str]) -> None:
    """Initialize the vector database with emails from all folders."""
    if not vector_db.is_emails_empty():
        typer.echo("Database already contains emails. Skipping initialization.")
        return

    typer.echo("Email database is empty. Downloading all emails...")
    for folder in folders:
        emails, all_uids = email_handler.get_mail(
            filter="all",
            folders=[folder],
            return_dataframe=True,
            return_uids=True
        )
        if not emails.empty:
            typer.echo(f"Storing {len(emails)} emails from {folder}")
            vector_db.store_emails(emails.to_dict(orient="records"))
        
        # Store all UIDs from the folder
        for folder_name, uids in all_uids.items():
            vector_db.add_seen_uids(uids, folder_name)

def get_vector_db(api_key: str = None) -> VectorDatabase:
    """Initialize and return the vector database."""
    try:
        config = read_config()
        email_db_path = os.path.expanduser(config["email_db_path"])
        
        # Create database directory if it doesn't exist
        os.makedirs(email_db_path, exist_ok=True)
        
        # Initialize vector database
        vector_db = VectorDatabase(
            db_path=email_db_path,
            embedding_function=config["default_embedding_function"],
            openai_api_key=api_key,
        )
        
        if vector_db.is_emails_empty():
            typer.secho(
                "Email database is empty. It will be initialized when emails are processed.",
                fg=typer.colors.YELLOW
            )
        
        return vector_db
    except Exception as e:
        typer.secho(
            f"Error initializing vector database: {str(e)}",
            err=True,
            fg=typer.colors.RED
        )
        raise
