import typer
from pathlib import Path
import shutil
from ..core.config_manager import read_config
from ..vector import VectorDatabase

database_app = typer.Typer(help="Manage email database")

@database_app.command("create")
def create_database(
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force creation even if database exists"
    )
) -> None:
    """Create a new email database."""
    try:
        config = read_config()
        db_path = Path(config["email_db_path"]).expanduser()

        if db_path.exists() and not force:
            typer.secho(
                f"Database already exists at {db_path}. Use --force to overwrite.",
                err=True,
                fg=typer.colors.RED
            )
            return

        # Create parent directory if it doesn't exist
        db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize empty database
        vector_db = VectorDatabase(
            db_path=str(db_path),
            embedding_function=config["default_embedding_function"]
        )
        
        typer.echo(f"✨ Created new email database at {db_path}")

    except Exception as e:
        typer.secho(
            f"Error creating database: {str(e)}",
            err=True,
            fg=typer.colors.RED
        )

@database_app.command("remove")
def remove_database(
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force removal without confirmation"
    )
) -> None:
    """Remove the email database."""
    try:
        config = read_config()
        db_path = Path(config["email_db_path"]).expanduser()

        if not db_path.exists():
            typer.secho(
                "Database does not exist.",
                err=True,
                fg=typer.colors.RED
            )
            return

        if not force:
            confirm = typer.confirm(
                "Are you sure you want to remove the email database? This action cannot be undone."
            )
            if not confirm:
                typer.echo("Operation cancelled.")
                return

        # Remove the database directory
        shutil.rmtree(db_path)
        typer.echo("✨ Email database removed successfully.")

    except Exception as e:
        typer.secho(
            f"Error removing database: {str(e)}",
            err=True,
            fg=typer.colors.RED
        )
