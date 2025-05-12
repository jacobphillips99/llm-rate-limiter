"""Rate limiting utilities for API calls with built-in monitoring."""

import logging
from typing import Any

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ModelRateLimitConfig(BaseModel):
    """Rate limit configuration for a specific model."""

    requests_per_minute: int = Field(default=0, ge=0)
    tokens_per_minute: int = Field(default=0, ge=0)

    class Config:
        """Pydantic config."""

        extra = "ignore"


class ProviderRateLimits(BaseModel):
    """Rate limits for all models of a provider."""

    models: dict[str, ModelRateLimitConfig] = Field(default_factory=dict)


class RateLimitConfig(BaseModel):
    """Top level rate limit configuration for all providers."""

    providers: dict[str, ProviderRateLimits] = Field(default_factory=dict)

    def get_provider_to_model_name(self) -> dict[str, list[str]]:
        """Get a mapping of provider to model name."""
        return {
            provider: list(self.providers[provider].models.keys())
            for provider in self.providers.keys()
        }


def load_yaml_config(file_path: str) -> dict[str, Any]:
    """Load YAML configuration file."""
    try:
        with open(file_path, "r") as f:
            config = yaml.safe_load(f)
        logger.debug(f"Loaded config from {file_path}")
        return config or {}
    except Exception as e:
        logger.warning(f"Error loading config from {file_path}: {str(e)}")
        return {}


def load_rate_limit_configs(config_path: str) -> RateLimitConfig:
    """Load rate limits from YAML configuration.

    Returns:
        A RateLimitConfig object containing rate limits for all providers and models.
        If no config file is found, returns an empty config.
    """
    yaml_data = load_yaml_config(config_path)
    # Convert the flat structure to our nested structure
    providers = {
        provider: ProviderRateLimits(models=models) for provider, models in yaml_data.items()
    }
    return RateLimitConfig(providers=providers)
