"""
Helper functions to load configuration from environment variables.
This prevents hardcoding sensitive information in the codebase.
"""

import os
from pathlib import Path
from typing import Optional

# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    # Load .env file from the same directory as this module
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    # python-dotenv not installed, skip auto-loading
    pass


def get_env_var(key: str, default: Optional[str] = None, required: bool = False) -> Optional[str]:
    """
    Get environment variable value.
    
    Args:
        key: Environment variable name
        default: Default value if not found
        required: If True, raise error if not found
        
    Returns:
        Environment variable value or default
        
    Raises:
        ValueError: If required=True and variable not found
    """
    value = os.getenv(key, default)
    if required and value is None:
        raise ValueError(f"Required environment variable '{key}' is not set")
    return value


def get_wandb_entity() -> str:
    """Get W&B entity from environment variable."""
    return get_env_var("WANDB_ENTITY", default="spirit-vilm")


def get_wandb_project() -> str:
    """Get W&B project name from environment variable."""
    return get_env_var("WANDB_PROJECT", default="hubert-vocoder")


def get_wandb_api_key() -> Optional[str]:
    """Get W&B API key from environment variable."""
    return get_env_var("WANDB_API_KEY")


def get_wandb_artifact_path(artifact_name: str, default_artifact: str) -> str:
    """
    Get W&B artifact path from environment variable.
    
    Args:
        artifact_name: Name of the artifact (e.g., 'hubert_model')
        default_artifact: Default artifact path if env var not set
        
    Returns:
        Artifact path from env var or default
    """
    env_key = f"WANDB_ARTIFACT_{artifact_name.upper()}"
    return get_env_var(env_key, default=default_artifact)

