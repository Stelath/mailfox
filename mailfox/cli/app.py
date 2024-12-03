import typer
from .credentials import credentials_app
from .config import config_app
from .commands import run

app = typer.Typer()

# Add sub-applications
app.add_typer(credentials_app, name="credentials")
app.add_typer(config_app, name="config")

# Add main command
@app.command()
def start():
    """Run the MailFox application."""
    run()