import typer
from typing import Optional
from .credentials import credentials_app
from .config import config_app
from .wizard import run_setup_wizard
from .run import run_application

app = typer.Typer(
    help="MailFox CLI - An intelligent email processing application",
    add_completion=False
)

# Add sub-applications
app.add_typer(
    credentials_app,
    name="credentials",
    help="Manage email and API credentials"
)
app.add_typer(
    config_app,
    name="config",
    help="Manage application configuration"
)

@app.command()
def start() -> None:
    """Start the MailFox application."""
    run_application()

@app.command()
def init() -> None:
    """Initialize MailFox with an interactive setup wizard."""
    run_setup_wizard()

@app.callback()
def main(
    verbose: Optional[bool] = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output"
    )
) -> None:
    """
    MailFox - Intelligent Email Processing

    Use 'init' to set up MailFox for first use.
    Use 'start' to begin processing emails.
    Use 'credentials' to manage your email and API credentials.
    Use 'config' to manage application settings.
    """
    if verbose:
        typer.echo("Verbose mode enabled")
