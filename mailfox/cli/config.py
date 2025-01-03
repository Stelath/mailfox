import typer
from typing import Optional
from pathlib import Path
import yaml
from ..core.config_manager import save_config, read_config
from ..vector import EmbeddingFunctions

config_app = typer.Typer(help="Manage MailFox configuration")


DEFAULT_CONFIG = {
    "default_classifier": "svm",
    "default_embedding_function": EmbeddingFunctions.SENTENCE_TRANSFORMER.value,
    "check_interval": 300,
    "enable_uid_validity": True,
    "classifier_model_path": None
}

@config_app.command("set")
def set_config(
    email_db_path: Optional[Path] = typer.Option(
        None,
        help="Path to email database",
        exists=False,
        file_okay=False,
        dir_okay=True,
        resolve_path=True
    ),
    clustering_path: Optional[Path] = typer.Option(
        None,
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
    default_classifier: Optional[str] = typer.Option(
        None,
        help="Default classifier (svm, logistic, clustering, or llm)"
    ),
    classifier_model_path: Optional[Path] = typer.Option(
        None,
        help="Path to the classifier model file",
        exists=False,
        file_okay=True,
        dir_okay=False,
        resolve_path=True
    ),
    default_embedding_function: Optional[EmbeddingFunctions] = typer.Option(
        None,
        help="Embedding function to use"
    ),
    check_interval: Optional[int] = typer.Option(
        None,
        help="Interval in seconds between email checks",
        min=60,
        max=3600
    ),
    enable_uid_validity: Optional[bool] = typer.Option(
        None,
        help="Enable UID validity checking"
    ),
) -> None:
    """Set the configuration for MailFox."""
    try:
        # Read existing config or start with empty dict
        try:
            config = read_config()
        except FileNotFoundError:
            config = {}
            
        # Update only specified values
        if email_db_path is not None:
            config["email_db_path"] = str(email_db_path)
        if clustering_path is not None:
            config["clustering_path"] = str(clustering_path)
        if flagged_folders is not None:
            config["flagged_folders"] = [folder.strip() for folder in flagged_folders.split(",")]
        if default_classifier is not None:
            config["default_classifier"] = default_classifier
        if classifier_model_path is not None:
            config["classifier_model_path"] = str(classifier_model_path)
        if default_embedding_function is not None:
            config["default_embedding_function"] = default_embedding_function.value
        if check_interval is not None:
            config["check_interval"] = check_interval
        if enable_uid_validity is not None:
            config["enable_uid_validity"] = enable_uid_validity
            
        # Ensure required values are present
        if "email_db_path" not in config:
            typer.secho("Error: email_db_path is required", err=True, fg=typer.colors.RED)
            raise typer.Exit(1)
        if "clustering_path" not in config:
            typer.secho("Error: clustering_path is required", err=True, fg=typer.colors.RED)
            raise typer.Exit(1)
        if "flagged_folders" not in config:
            config["flagged_folders"] = ["INBOX"]
            
        # Set defaults for unspecified values only if they don't exist
        for key, value in DEFAULT_CONFIG.items():
            if key not in config:
                config[key] = value
        
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

@config_app.command("reset")
def reset_config() -> None:
    """Reset configuration to defaults, preserving paths and folders."""
    try:
        config = read_config()
        
        # Keep only paths and folders
        preserved = {
            "email_db_path": config["email_db_path"],
            "clustering_path": config["clustering_path"],
            "flagged_folders": config["flagged_folders"]
        }
        
        # Add defaults
        config = {**preserved, **DEFAULT_CONFIG}
        
        save_config(config)
        typer.echo("✨ Configuration reset successfully.")
        
    except Exception as e:
        typer.secho(
            f"Error resetting configuration: {str(e)}",
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
            
        typer.echo("✅ Configuration validation complete")
        
    except Exception as e:
        typer.secho(
            f"Error validating configuration: {str(e)}",
            err=True,
            fg=typer.colors.RED
        )
