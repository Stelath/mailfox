import typer
from ..core.config_manager import save_config, read_config
import yaml

config_app = typer.Typer()

@config_app.command("set")
def set_config(
    email_db_path: str = typer.Option(..., help="Path to email database"),
    clustering_path: str = typer.Option(..., help="Path to clustering data"),
    flagged_folders: str = typer.Option(
        None, help="Comma-separated list of folders to flag (optional)"
    ),
    default_classifier: str = typer.Option(
        "clustering", help="Default classifier (clustering or llm)"
    ),
    default_embedding_function: str = typer.Option(
        "st", help="Embedding function (st or openai)"
    ),
):
    """Set the configuration for the application."""
    config = {
        "email_db_path": email_db_path,
        "clustering_path": clustering_path,
        "flagged_folders": (
            [folder.strip() for folder in flagged_folders.split(",")]
            if flagged_folders
            else []
        ),
        "default_classifier": default_classifier,
        "default_embedding_function": default_embedding_function,
    }
    
    save_config(config)
    typer.echo("Configuration saved successfully.")

@config_app.command("list")
def list_config():
    """List the current configuration."""
    try:
        config = read_config()
        typer.echo(yaml.dump(config, default_flow_style=False))
    except FileNotFoundError:
        typer.secho("No configuration file found.", err=True, fg=typer.colors.RED)