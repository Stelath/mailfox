import typer
import os
from ..core.auth import remove_credentials_file, read_credentials, save_credentials
from typing import Optional

credentials_app = typer.Typer()

@credentials_app.command("set")
def set_credentials(
    username: str = typer.Argument(..., help="Your email address"),
    password: str = typer.Argument(..., help="Your email password", hidden=True),
    api_key: Optional[str] = typer.Argument(None, help="Your OpenAI API key (optional)", hidden=True)
):
    """Save email and API credentials."""
    save_credentials(username, password, api_key)
    typer.echo("Credentials saved successfully.")

@credentials_app.command("remove")
def remove_credentials():
    """Remove stored email credentials."""
    remove_credentials_file()
    typer.echo("Credentials removed.")

@credentials_app.command("show")
def show_credentials():
    """Display saved credentials (except the password)."""
    try:
        username, _, api_key = read_credentials()
        typer.echo(f"username={username}")
        if api_key:
            typer.echo(f"api_key={api_key}")
    except FileNotFoundError:
        typer.secho("No credentials found.", err=True, fg=typer.colors.RED)