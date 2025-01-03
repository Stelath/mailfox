import typer
import os
from typing import Callable, Optional, Set
import pandas as pd
from ..core.email_processor import process_new_mail, initialize_classifier
from ..core.auth import read_credentials
from ..core.config_manager import read_config
from ..core.database_manager import get_vector_db, initialize_database
from ..email_interface import EmailHandler
from ..vector import VectorDatabase

def process_folder_update(
    folder: str,
    emails: pd.DataFrame,
    vector_db: VectorDatabase,
    recache: bool = False,
    fetched_uids: Optional[Set[int]] = None,
    current_uids: Optional[Set[int]] = None
) -> None:
    """Process updates in a monitored folder."""
    try:
        # Store any fetched UIDs
        if fetched_uids:
            vector_db.add_seen_uids(fetched_uids, folder)
        elif current_uids:  # For new folders
            vector_db.add_seen_uids(current_uids, folder)

        if not emails.empty:
            if recache:
                typer.echo(f"Recaching {len(emails)} emails in {folder}")
                vector_db.store_emails(emails.to_dict(orient="records"))
            else:
                typer.echo(f"Processing {len(emails)} new emails in {folder}")
                vector_db.store_emails(emails.to_dict(orient="records"))
    except Exception as e:
        typer.secho(
            f"Error processing folder {folder} update: {str(e)}",
            err=True,
            fg=typer.colors.RED
        )

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
            
        # Initialize classifier if needed
        model_path = config.get('classifier_model_path')
        if not model_path:
            model_path = os.path.expanduser(config['clustering_path'])
        else:
            model_path = os.path.expanduser(model_path)
            
        if not os.path.exists(model_path):
            typer.echo(f"Initializing new {config['default_classifier']} classifier...")
            initialize_classifier(vector_db)
            typer.echo("Classifier initialized and saved.")
        else:
            typer.echo(f"Using existing {config['default_classifier']} classifier.")
            
        # Get all folders including subfolders
        classification_folders = config['flagged_folders']
        all_folders = email_handler.get_subfolders(classification_folders)
        typer.echo(f"Monitoring folders: {', '.join(all_folders)}")
        
        # Initialize database if empty
        initialize_database(email_handler, vector_db, all_folders)
        
        # Initialize folder UIDs from seen UIDs in database
        folder_uids = vector_db.get_seen_uids()
        
        # Start monitoring for new emails and UID validity changes
        check_interval = config.get("check_interval", 300)
        enable_uid_validity = config.get("enable_uid_validity", True)
        
        typer.echo(f"Starting email monitoring (checking every {check_interval} seconds)")
        try:
            while not email_handler.stop_event.is_set():
                # First check inbox
                process_new_mail("INBOX", email_handler, vector_db)
                
                # Then poll folders
                email_handler.poll_folders(
                    folders=all_folders,
                    folder_uids=folder_uids,
                    callback=lambda folder, emails, recache=False, fetched_uids=None, current_uids=None: process_folder_update(
                        folder, emails, vector_db, recache, fetched_uids, current_uids
                    ),
                    enable_uid_validity=enable_uid_validity
                )
                
                # Wait before next iteration
                email_handler.stop_event.wait(check_interval)
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
