"""
MailFox CLI Module

This module provides the command-line interface for MailFox, an intelligent email
processing application. The CLI is built using Typer and provides commands for:

- Initializing the application with a setup wizard
- Starting the email processing service
- Managing credentials (email and API keys)
- Configuring application settings

The CLI structure is organized into several submodules:
- app.py: Main application entry point and command registration
- wizard.py: Interactive setup wizard implementation
- run.py: Core application runtime logic
- config.py: Configuration management commands
- credentials.py: Credential management commands
"""

from .app import app

__all__ = ['app']
