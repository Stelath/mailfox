import typer
from pathlib import Path
import shutil
from ..core.config_manager import read_config
from ..core.auth import read_credentials
from ..vector import VectorDatabase, EmbeddingFunctions
from ..vector.classifiers.linear_svm import LinearSVMClassifier
from ..vector.classifiers.logistic_regression import LogisticRegressionClassifier
import numpy as np

database_app = typer.Typer(help="Manage email database")

CLASSIFIERS = {
    "svm": LinearSVMClassifier,
    "logistic": LogisticRegressionClassifier
}

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

        openai_api_key = None
        if config["default_embedding_function"] == EmbeddingFunctions.OPENAI:
            # Try to get API key from credentials
            try:
                _, _, api_key = read_credentials()
                openai_api_key = api_key
            except Exception as e:
                print(f"Error reading credentials: {e}")
                raise typer.Exit(1)

            if not openai_api_key:
                typer.echo("OpenAI API key is required for OpenAI embeddings. Set it using 'mailfox credentials set'")
                raise typer.Exit(1)

        # Initialize empty database
        vector_db = VectorDatabase(
            db_path=str(db_path),
            embedding_function=config["default_embedding_function"],
            openai_api_key=openai_api_key
        )
        
        typer.echo(f"✨ Created new email database at {db_path}")

    except Exception as e:
        typer.secho(
            f"Error creating database: {str(e)}",
            err=True,
            fg=typer.colors.RED
        )

@database_app.command("retrain")
def retrain_classifier(
    classifier: str = typer.Option(
        "svm",
        "--classifier",
        "-c",
        help="Classifier to train (svm or logistic)"
    ),
    model_path: Path = typer.Option(
        None,
        "--model-path",
        "-m",
        help="Path to save the trained model"
    )
) -> None:
    """Retrain a classifier using the current email database."""
    try:
        if classifier not in CLASSIFIERS:
            typer.secho(
                f"Invalid classifier. Choose from: {', '.join(CLASSIFIERS.keys())}",
                err=True,
                fg=typer.colors.RED
            )
            return

        # Read configuration
        config = read_config()
        db_path = Path(config["email_db_path"]).expanduser()

        # Get API key if needed
        openai_api_key = None
        if config["default_embedding_function"] == EmbeddingFunctions.OPENAI:
            try:
                _, _, api_key = read_credentials()
                openai_api_key = api_key
            except Exception as e:
                typer.secho(f"Error reading credentials: {e}", err=True, fg=typer.colors.RED)
                return

        # Initialize vector database
        vector_db = VectorDatabase(
            db_path=str(db_path),
            embedding_function=config["default_embedding_function"],
            openai_api_key=openai_api_key
        )

        # Get all emails and their embeddings
        emails = vector_db.get_all_emails()
        if not emails:
            typer.secho("No emails found in database", err=True, fg=typer.colors.RED)
            return

        # Get embeddings for all emails
        all_embeddings = []
        all_folders = []
        for email in emails:
            embeddings = vector_db.embed_paragraphs(email['paragraphs'])
            if embeddings:
                all_embeddings.extend(embeddings)
                all_folders.extend([email['folder']] * len(embeddings))

        if not all_embeddings:
            typer.secho("No embeddings found", err=True, fg=typer.colors.RED)
            return

        # Convert embeddings to numpy array
        embeddings_array = np.array(all_embeddings)

        # Initialize and train classifier
        clf = CLASSIFIERS[classifier]()
        clf.fit(embeddings_array, all_folders)

        # Save model
        if model_path is None:
            model_path = Path(config["clustering_path"]).parent / f"{classifier}_model.pkl"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        clf.save_model(model_path)

        typer.echo(f"✨ Successfully trained {classifier} classifier and saved to {model_path}")

    except Exception as e:
        typer.secho(
            f"Error training classifier: {str(e)}",
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
