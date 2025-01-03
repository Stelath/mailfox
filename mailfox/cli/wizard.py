import typer
import click
from typing import Optional, List
from ..core.auth import save_credentials
from ..core.config_manager import save_config
from ..core.database_manager import get_vector_db, initialize_database
from ..email_interface import EmailHandler
from ..vector import EmbeddingFunctions
import os

def run_setup_wizard() -> None:
    """Run the interactive setup wizard for MailFox configuration."""
    typer.echo("Welcome to the setup wizard!")
    
    # Step 1: Credentials
    typer.echo("\nStep 1: Setting your credentials")
    username = typer.prompt("Enter your email address")
    password = typer.prompt("Enter your email password", hide_input=True)
    api_key = typer.prompt("Enter your API key (optional)", default="", hide_input=True)
    save_credentials(username, password, api_key)
    
    # Step 2: Paths and Configuration
    typer.echo("\nStep 2: Configuring paths")
    config = _configure_paths_and_settings()
    save_config(config)
    
    # Step 3: Initial Email Download
    typer.echo("\nStep 3: Downloading emails")
    _handle_initial_download(username, password, api_key, config)
    
    typer.echo("\n✨ Setup complete! You can now run 'mailfox start' to begin processing emails.")

def _configure_paths_and_settings() -> dict:
    """Configure paths and settings interactively."""
    email_db_path = typer.prompt(
        "Enter the path for email database",
        default="~/.mailfox/data/email_db"
    )
    clustering_path = typer.prompt(
        "Enter the path for clustering data",
        default="~/.mailfox/data/clustering.pkl"
    )
    flagged_folders = typer.prompt(
        "Enter any flagged folders (comma-separated)",
        default="INBOX"
    )
    default_classifier = typer.prompt(
        "Enter the default classifier",
        default="clustering"
    )
    
    # Use Click's Choice for validated input
    embedding_choices = [func.value for func in EmbeddingFunctions]
    default_embedding_function = typer.prompt(
        "Enter the default embedding function",
        default="st",
        type=click.Choice(embedding_choices)
    )
    
    check_interval = typer.prompt(
        "Enter check interval in seconds",
        default=300,
        type=int
    )
    enable_uid_validity = typer.confirm(
        "Enable UID validity checking?",
        default=True
    )
    
    return {
        "email_db_path": email_db_path,
        "clustering_path": clustering_path,
        "flagged_folders": [f.strip() for f in flagged_folders.split(",")],
        "default_classifier": default_classifier,
        "default_embedding_function": default_embedding_function,
        "check_interval": check_interval,
        "enable_uid_validity": enable_uid_validity,
        "recache_limit": recache_limit,
    }

def _handle_initial_download(
    username: str,
    password: str,
    api_key: Optional[str],
    config: dict
) -> None:
    """Handle the initial email download process."""
    should_download = typer.confirm(
        "Would you like to download your emails to the VectorDB now?"
    )
    if not should_download:
        typer.echo("Email download skipped.")
        return
        
    try:
        email_handler = EmailHandler(username, password)
        vector_db = get_vector_db(api_key)
        
        folders = config["flagged_folders"]
        typer.echo(f"Downloading emails from folders: {', '.join(folders)}")
        
        initialize_database(email_handler, vector_db, folders)
                
    except Exception as e:
        typer.secho(
            f"Error during initial download: {str(e)}",
            err=True,
            fg=typer.colors.RED
        )
        typer.echo(
            "You can try downloading emails later using 'mailfox download' command."
        )
