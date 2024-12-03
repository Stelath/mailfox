import typer
import os
from ..core.email_processor import process_new_mail, initialize_clustering
from ..core.auth import read_credentials
from ..core.config_manager import read_config
from ..email_interface import EmailHandler
from ..vector import VectorDatabase

def run():
    """Run the application."""
    try:
        # Retrieve credentials and config
        username, password, api_key = read_credentials()
        config = read_config()
        email_handler = EmailHandler(username, password)
        vector_db = get_vector_db(api_key=api_key, email_handler=email_handler)

        if vector_db is None:
            return

        # Initialize and save new clustering model
        typer.echo("Initializing new clustering model...")
        initialize_clustering(vector_db)
        typer.echo("Clustering model initialized and saved.")

        # Check for unread emails in INBOX before entering IDLE mode
        typer.echo("Checking for unread emails in INBOX...")
        unread_emails = email_handler.get_mail(filter="unseen", folders=["INBOX"], return_dataframe=True)
        if not unread_emails.empty:
            typer.echo(f"Found {len(unread_emails)} unread emails.")
            vector_db.store_emails(unread_emails.to_dict(orient="records"))
            process_new_mail("INBOX", email_handler, vector_db)
        else:
            typer.echo("No unread emails in INBOX.")

        # Start IDLE mode to monitor folders
        typer.echo("Entering IMAP IDLE mode to monitor email folders...")
        folders_to_monitor = config.get("flagged_folders", ["INBOX"])
        email_handler.idle_check_new_mail(
            folders=folders_to_monitor,
            callback=lambda folder: process_new_mail(folder, email_handler, vector_db),
        )
    except Exception as e:
        typer.secho(f"Error running application: {e}", err=True, fg=typer.colors.RED)

def get_vector_db(api_key, *, email_handler):
    config = read_config()
    email_db_path = os.path.expanduser(config["email_db_path"])
    if os.path.exists(email_db_path) and not VectorDatabase(email_db_path).is_emails_empty():
        vector_db = VectorDatabase(
            db_path=email_db_path,
            embedding_function=config["default_embedding_function"],
            openai_api_key=api_key,
        )
    else:
        typer.secho("No email database found. Please initialize the database first.", err=True, fg=typer.colors.RED)
        return None
    return vector_db