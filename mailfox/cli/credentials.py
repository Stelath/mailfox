import typer
from typing import Optional
from pathlib import Path
from ..core.auth import remove_credentials_file, read_credentials, save_credentials

credentials_app = typer.Typer(help="Manage email and API credentials")

@credentials_app.command("set")
def set_credentials(
    username: str = typer.Argument(
        ...,
        help="Your email address"
    ),
    password: str = typer.Option(
        ...,
        prompt=True,
        hide_input=True,
        help="Your email password"
    ),
    api_key: Optional[str] = typer.Option(
        None,
        help="Your OpenAI API key",
        prompt="Enter your OpenAI API key (optional)",
        hide_input=True
    )
) -> None:
    """Set your email and API credentials."""
    try:
        save_credentials(username, password, api_key)
        typer.echo("✨ Credentials saved successfully.")
    except Exception as e:
        typer.secho(
            f"Error saving credentials: {str(e)}",
            err=True,
            fg=typer.colors.RED
        )

@credentials_app.command("remove")
def remove_credentials(
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force removal without confirmation"
    )
) -> None:
    """Remove stored credentials."""
    try:
        if not force:
            confirm = typer.confirm(
                "Are you sure you want to remove your stored credentials?"
            )
            if not confirm:
                typer.echo("Operation cancelled.")
                return
                
        remove_credentials_file()
        typer.echo("✨ Credentials removed successfully.")
    except Exception as e:
        typer.secho(
            f"Error removing credentials: {str(e)}",
            err=True,
            fg=typer.colors.RED
        )

@credentials_app.command("show")
def show_credentials() -> None:
    """Display saved credentials (excluding sensitive data)."""
    try:
        username, _, api_key = read_credentials()
        
        typer.echo("\nStored Credentials:")
        typer.echo("==================")
        typer.echo(f"Email: {username}")
        if api_key:
            # Show only last 4 characters of API key
            masked_key = f"...{api_key[-4:]}" if len(api_key) > 4 else "****"
            typer.echo(f"API Key: {masked_key}")
        else:
            typer.echo("API Key: Not set")
            
    except FileNotFoundError:
        typer.secho(
            "No credentials found. Run 'mailfox credentials set' to set up your credentials.",
            err=True,
            fg=typer.colors.RED
        )
    except Exception as e:
        typer.secho(
            f"Error reading credentials: {str(e)}",
            err=True,
            fg=typer.colors.RED
        )

@credentials_app.command("validate")
def validate_credentials() -> None:
    """Validate stored credentials."""
    try:
        username, password, api_key = read_credentials()
        
        # Basic validation
        if not username or not password:
            typer.secho(
                "Error: Email credentials are incomplete.",
                err=True,
                fg=typer.colors.RED
            )
            return
            
        # Validate email format
        if "@" not in username or "." not in username:
            typer.secho(
                f"Warning: Email address '{username}' may not be valid.",
                fg=typer.colors.YELLOW
            )
            
        # Check API key format if present
        if api_key:
            if len(api_key) < 20:  # Arbitrary minimum length for API keys
                typer.secho(
                    "Warning: API key seems too short.",
                    fg=typer.colors.YELLOW
                )
                
        typer.echo("✅ Credentials validation complete")
        
    except FileNotFoundError:
        typer.secho(
            "No credentials found. Run 'mailfox credentials set' to set up your credentials.",
            err=True,
            fg=typer.colors.RED
        )
    except Exception as e:
        typer.secho(
            f"Error validating credentials: {str(e)}",
            err=True,
            fg=typer.colors.RED
        )
