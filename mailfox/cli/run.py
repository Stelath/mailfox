import typer
import os
from typing import Callable, Optional
import pandas as pd
from ..core.email_processor import process_new_mail, initialize_clustering
from ..core.auth import read_credentials
from ..core.config_manager import read_config
from ..email_interface import EmailHandler
from ..vector import VectorDatabase

def process_folder_update(
    folder: str,
    emails: pd.DataFrame,
    email_handler: EmailHandler,
    vector_db: VectorDatabase,
    recache: bool = False
) -> None:
    """Process updates in a monitored folder."""
    try:
        if not emails.empty:
            action = "Recaching" if recache else "Processing"
            typer.echo(f"{action} {len(emails)} emails in {folder}")
            vector_db.store_emails(emails.to_dict(orient="records"))
            if not recache:
                process_new_mail(folder, email_handler, vector_db)
    except Exception as e:
        typer.secho(
            f"Error processing folder {folder} update: {str(e)}",
            err=True,
            fg=typer.colors.RED
        )

def get_vector_db(api_key: Optional[str]) -> Optional[VectorDatabase]:
    """Initialize and return the vector database."""
    try:
        config = read_config()
        email_db_path = os.path.expanduser(config["email_db_path"])
        
        if os.path.exists(email_db_path) and not VectorDatabase(email_db_path).is_emails_empty():
            return VectorDatabase(
                db_path=email_db_path,
                embedding_function=config["default_embedding_function"],
                openai_api_key=api_key,
            )
        else:
            typer.secho(
                "No email database found. Please initialize the database first.",
                err=True,
                fg=typer.colors.RED
            )
            return None
    except Exception as e:
        typer.secho(
            f"Error initializing vector database: {str(e)}",
            err=True,
            fg=typer.colors.RED
        )
        return None

def run_application() -> None:
    """Run the main MailFox application."""
    email_handler = None
    try:
        # Initialize components
        username, password, api_key = read_credentials()
        config = read_config()
        email_handler = EmailHandler(username, password)
        vector_db = get_vector_db(api_key)
        
        if vector_db is None:
            return
            
        # Initialize clustering if needed
        config_path = os.path.expanduser(config['clustering_path'])
        if not os.path.exists(config_path):
            typer.echo("Initializing new clustering model...")
            initialize_clustering(vector_db)
            typer.echo("Clustering model initialized and saved.")
        else:
            typer.echo("Using existing clustering model.")
            
        # Start monitoring
        folders_to_monitor = config.get("flagged_folders", ["INBOX"])
        check_interval = config.get("check_interval", 300)
        enable_uid_validity = config.get("enable_uid_validity", True)
        recache_limit = config.get("recache_limit", 100)
        
        typer.echo(f"Starting email monitoring (checking every {check_interval} seconds)")
        try:
            email_handler.poll_folders(
                folders=folders_to_monitor,
                callback=lambda folder, emails, recache=False: process_folder_update(
                    folder, emails, email_handler, vector_db, recache
                ),
                check_interval=check_interval,
                enable_uid_validity=enable_uid_validity,
                recache_limit=recache_limit
            )
        except KeyboardInterrupt:
            typer.echo("\nShutting down gracefully...")
        finally:
            if email_handler:
                email_handler.stop_event.set()
                
    except Exception as e:
        typer.secho(
            f"Error running application: {str(e)}",
            err=True,
            fg=typer.colors.RED
        )
        if email_handler:
            email_handler.stop_event.set()