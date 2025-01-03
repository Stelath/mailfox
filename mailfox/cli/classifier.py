import typer
from pathlib import Path
import os
from typing import Optional
from ..core.config_manager import read_config
from ..core.auth import read_credentials
from ..vector import VectorDatabase, EmbeddingFunctions
from ..vector.classifiers.linear_svm import LinearSVMClassifier
from ..vector.classifiers.logistic_regression import LogisticRegressionClassifier
import numpy as np

classifier_app = typer.Typer(help="Manage email classifiers")

CLASSIFIERS = {
    "svm": LinearSVMClassifier,
    "logistic": LogisticRegressionClassifier
}

def format_metrics(metrics: dict) -> str:
    """Format metrics for display."""
    if not metrics:
        return "No metrics available"
        
    return f"""
Metrics:
  • Accuracy: {metrics['accuracy']:.2%}
  • Precision: {metrics['precision']:.2%}
  • Recall: {metrics['recall']:.2%}
  • F1 Score: {metrics['f1']:.2%}
  • Training Data:
    - Total Samples: {metrics['num_samples']}
    - Number of Folders: {metrics['num_folders']}
"""

@classifier_app.command("show")
def show_classifier() -> None:
    """Display information about the current classifier."""
    try:
        config = read_config()
        classifier_type = config.get('default_classifier', 'svm')
        
        if classifier_type not in CLASSIFIERS:
            typer.secho(
                f"Invalid classifier type: {classifier_type}",
                err=True,
                fg=typer.colors.RED
            )
            return
            
        model_path = config.get('classifier_model_path')
        if not model_path:
            model_path = os.path.expanduser(config['clustering_path'])
        else:
            model_path = os.path.expanduser(model_path)
            
        if not os.path.exists(model_path):
            typer.secho(
                "No trained classifier found.",
                err=True,
                fg=typer.colors.RED
            )
            return
            
        # Load classifier and get metrics
        classifier = CLASSIFIERS[classifier_type]()
        metrics = classifier.load_model(model_path)
        
        typer.echo("\nCurrent Classifier:")
        typer.echo("==================")
        typer.echo(f"Type: {classifier_type}")
        typer.echo(f"Model Path: {model_path}")
        typer.echo(format_metrics(metrics))
        
    except Exception as e:
        typer.secho(
            f"Error showing classifier: {str(e)}",
            err=True,
            fg=typer.colors.RED
        )

@classifier_app.command("delete")
def delete_classifier(
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force deletion without confirmation"
    )
) -> None:
    """Delete the current classifier."""
    try:
        config = read_config()
        model_path = config.get('classifier_model_path')
        if not model_path:
            model_path = os.path.expanduser(config['clustering_path'])
        else:
            model_path = os.path.expanduser(model_path)
            
        if not os.path.exists(model_path):
            typer.secho(
                "No classifier found to delete.",
                err=True,
                fg=typer.colors.RED
            )
            return
            
        if not force:
            confirm = typer.confirm(
                "Are you sure you want to delete the classifier?"
            )
            if not confirm:
                typer.echo("Operation cancelled.")
                return
                
        os.remove(model_path)
        typer.echo("✨ Classifier deleted successfully.")
        
    except Exception as e:
        typer.secho(
            f"Error deleting classifier: {str(e)}",
            err=True,
            fg=typer.colors.RED
        )

@classifier_app.command("retrain")
def retrain_classifier(
    classifier_type: Optional[str] = typer.Option(
        None,
        "--type",
        "-t", 
        help="Type of classifier to train (svm or logistic)"
    ),
    model_path: Optional[Path] = typer.Option(
        None,
        "--model-path",
        "-m",
        help="Path to save the trained model"
    )
) -> None:
    """Retrain the classifier using the current email database."""
    try:
        config = read_config()
        
        # Use config values if not specified 
        if classifier_type is None:
            classifier_type = config.get('default_classifier', 'svm')
            
        if classifier_type not in CLASSIFIERS:
            typer.secho(
                f"Invalid classifier type. Choose from: {', '.join(CLASSIFIERS.keys())}",
                err=True,
                fg=typer.colors.RED
            )
            return
            
        if model_path is None:
            model_path = config.get('classifier_model_path')
            if not model_path:
                model_path = os.path.expanduser(config['clustering_path'])
            else:
                model_path = os.path.expanduser(model_path)

        # Initialize vector database
        typer.echo("🔌 Connecting to vector database...")
        db_path = Path(config["email_db_path"]).expanduser()
        openai_api_key = None
        if config["default_embedding_function"] == EmbeddingFunctions.OPENAI:
            typer.echo("🔑 Loading OpenAI credentials...")
            try:
                _, _, api_key = read_credentials()
                openai_api_key = api_key
                typer.echo("✅ OpenAI credentials loaded successfully")
            except Exception as e:
                typer.secho(f"Error reading credentials: {e}", err=True, fg=typer.colors.RED)
                return

        vector_db = VectorDatabase(
            db_path=str(db_path),
            embedding_function=config["default_embedding_function"],
            openai_api_key=openai_api_key
        )

        # Get existing embeddings from database
        typer.echo("📊 Loading embeddings from database...")
        docs = vector_db.emails_collection.get(include=['embeddings', 'metadatas'])
        
        if not docs['embeddings']:
            typer.secho("❌ No embeddings found in database", err=True, fg=typer.colors.RED)
            return
            
        embeddings_array = np.array(docs['embeddings'])
        folders = [metadata['folder'] for metadata in docs['metadatas']]
        
        typer.echo(f"✨ Loaded {len(embeddings_array)} embeddings from {len(set(folders))} folders")

        # Initialize and train classifier
        clf = CLASSIFIERS[classifier_type]()
        with typer.progressbar(
            length=100,
            label=f"🧠 Training {classifier_type} classifier"
        ) as progress:
            metrics = clf.fit(embeddings_array, folders)
            progress.update(100)

        # Save model with metrics
        typer.echo("\n💾 Saving trained model...")
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        clf.save_model(model_path, metrics)

        typer.echo(f"\n🎉 Successfully trained {classifier_type} classifier and saved to {model_path}")
        typer.echo(format_metrics(metrics))

    except Exception as e:
        typer.secho(
            f"❌ Error training classifier: {str(e)}",
            err=True,
            fg=typer.colors.RED
        )
