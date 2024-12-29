import typer
from typing import Optional
from pathlib import Path
import yaml
from ..core.config_manager import save_config, read_config
from ..vector import EmbeddingFunctions

config_app = typer.Typer(help="Manage MailFox configuration")


@config_app.command("set")
def set_config(
    email_db_path: Path = typer.Option(
        ...,
        help="Path to email database",
        exists=False,
        file_okay=False,
        dir_okay=True,
        resolve_path=True
    ),
    clustering_path: Path = typer.Option(
        ...,
        help="Path to clustering data",
        exists=False,
        file_okay=True,
        dir_okay=False,
        resolve_path=True
    ),
    flagged_folders: Optional[str] = typer.Option(
        None,
        help="Comma-separated list of folders to monitor"
    ),
    default_classifier: str = typer.Option(
        "clustering",
        help="Default classifier (clustering or llm)"
    ),
    default_embedding_function: EmbeddingFunctions = typer.Option(
        EmbeddingFunctions.SENTENCE_TRANSFORMER,
        help="Embedding function to use"
    ),
    check_interval: int = typer.Option(
        300,
        help="Interval in seconds between email checks",
        min=60,
        max=3600
    ),
    enable_uid_validity: bool = typer.Option(
        True,
        help="Enable UID validity checking"
    ),
    recache_limit: int = typer.Option(
        100,
        help="Number of emails to recache on UID validity mismatch",
        min=1,
        max=1000
    ),
) -> None:
    """Set the configuration for MailFox."""
    try:
        config = {
            "email_db_path": str(email_db_path),
            "clustering_path": str(clustering_path),
            "flagged_folders": (
                [folder.strip() for folder in flagged_folders.split(",")]
                if flagged_folders
                else ["INBOX"]
            ),
            "default_classifier": default_classifier,
            "default_embedding_function": default_embedding_function.value,
            "check_interval": check_interval,
            "enable_uid_validity": enable_uid_validity,
            "recache_limit": recache_limit,
        }
        
        save_config(config)
        typer.echo("✨ Configuration saved successfully.")
        
    except Exception as e:
        typer.secho(
            f"Error saving configuration: {str(e)}",
            err=True,
            fg=typer.colors.RED
        )

@config_app.command("show")
def show_config() -> None:
    """Display the current configuration."""
    try:
        config = read_config()
        typer.echo("\nCurrent Configuration:")
        typer.echo("=====================")
        typer.echo(yaml.dump(config, default_flow_style=False, sort_keys=False))
        
    except FileNotFoundError:
        typer.secho(
            "No configuration file found. Run 'mailfox init' to set up MailFox.",
            err=True,
            fg=typer.colors.RED
        )
    except Exception as e:
        typer.secho(
            f"Error reading configuration: {str(e)}",
            err=True,
            fg=typer.colors.RED
        )

@config_app.command("validate")
def validate_config() -> None:
    """Validate the current configuration."""
    try:
        config = read_config()
        
        # Validate paths
        email_db_path = Path(config["email_db_path"]).expanduser()
        clustering_path = Path(config["clustering_path"]).expanduser()
        
        # Check parent directories exist
        if not email_db_path.parent.exists():
            typer.secho(
                f"Warning: Directory for email database does not exist: {email_db_path.parent}",
                fg=typer.colors.YELLOW
            )
            
        if not clustering_path.parent.exists():
            typer.secho(
                f"Warning: Directory for clustering data does not exist: {clustering_path.parent}",
                fg=typer.colors.YELLOW
            )
            
        # Validate embedding function
        if config["default_embedding_function"] not in [e.value for e in EmbeddingFunctions]:
            typer.secho(
                f"Error: Invalid embedding function: {config['default_embedding_function']}",
                fg=typer.colors.RED
            )
            return
            
        # Validate numeric values
        if not (60 <= config["check_interval"] <= 3600):
            typer.secho(
                f"Warning: Check interval {config['check_interval']} is outside recommended range (60-3600)",
                fg=typer.colors.YELLOW
            )
            
        if not (1 <= config["recache_limit"] <= 1000):
            typer.secho(
                f"Warning: Recache limit {config['recache_limit']} is outside recommended range (1-1000)",
                fg=typer.colors.YELLOW
            )
            
        typer.echo("✅ Configuration validation complete")
        
    except Exception as e:
        typer.secho(
            f"Error validating configuration: {str(e)}",
            err=True,
            fg=typer.colors.RED
        )
