import asyncio
import json
import logging
import os
import threading
import time
from typing import Any, Optional

from llm_rate_limiter.configs import ModelRateLimitConfig, RateLimitConfig, load_rate_limit_configs
from llm_rate_limiter.constants import (
    DEFAULT_RATE_LIMIT_CONFIG_PATH,
    RATE_LIMIT_STATS_PATH,
    RATE_LIMIT_WRITE_INTERVAL,
)

logger = logging.getLogger(__name__)


class RateLimit:
    """Rate limit handler for API calls with built-in monitoring."""

    def __init__(
        self,
        stats_path: str = RATE_LIMIT_STATS_PATH,
        monitor_interval: float = RATE_LIMIT_WRITE_INTERVAL,
        disable_monitoring: bool = False,
        config: Optional[RateLimitConfig] = None,
    ) -> None:
        """Initialize the rate limit handler.

        Args:
            stats_path: Path to write stats to. If None, uses /tmp/rate_limit_stats.json
            monitor_interval: How often to update the stats file (seconds)
            disable_monitoring: If True, won't write stats to file
            config: Rate limit configuration
        """
        self._provider_model_configs: dict[str, dict[str, ModelRateLimitConfig]] = {}
        self._provider_model_states: dict[str, dict[str, dict]] = {}
        self._locks: dict[str, dict[str, threading.Lock]] = {}

        # Stats monitoring
        self._stats_path = stats_path
        self._monitor_interval = monitor_interval
        self._monitor_running = not disable_monitoring
        self._monitor_task: Optional[threading.Thread] = None
        self._monitor_lock = threading.Lock()

        # Start monitoring automatically if not disabled
        if self._monitor_running:
            # This will be initialized when register_model is first called
            # or when get_usage_stats is first called
            pass

        self._config = config
        if self._config:
            self.register_config(self._config)

    def _ensure_monitor_started(self) -> None:
        """Ensure the monitoring task is started if enabled."""
        if not self._monitor_running:
            return

        # Use a class-level lock to ensure thread safety
        with self._monitor_lock:
            if self._monitor_task is not None:
                return

            try:

                def run_monitor() -> None:
                    while self._monitor_running:
                        try:
                            stats = {
                                "timestamp": time.time(),
                                "stats": self.get_usage_stats(),
                            }
                            # Ensure directory exists
                            stats_dir = os.path.dirname(self._stats_path)
                            os.makedirs(stats_dir, exist_ok=True)

                            with open(self._stats_path, "w") as f:
                                json.dump(stats, f, indent=2)
                                f.flush()
                                os.fsync(f.fileno())

                        except Exception as e:
                            logger.error(f"Error writing stats to {self._stats_path}: {e}")
                        time.sleep(self._monitor_interval)

                thread = threading.Thread(target=run_monitor, daemon=True)
                thread.start()
                self._monitor_task = thread
                logger.info(
                    f"Started rate limit monitoring at {time.time()}, writing to {self._stats_path} every {self._monitor_interval}s"
                )
            except Exception as e:
                logger.error(f"Failed to start monitoring: {e}")
                self._monitor_running = False

    @property
    def providers(self) -> list[str]:
        """Get list of registered providers.

        Returns:
            List of provider names that have been registered
        """
        return list(self._provider_model_configs.keys())

    @property
    def providers_to_models(self) -> dict[str, list[str]]:
        """Get a dictionary of providers to their registered models.

        Returns:
            Dictionary of provider names to lists of model names
        """
        return {
            provider: list(self._provider_model_configs[provider].keys())
            for provider in self.providers
        }

    def register_model(self, provider: str, model: str, config: ModelRateLimitConfig) -> None:
        """Register a model with its rate limit configuration.

        Args:
            provider: The provider name (e.g., "openai")
            model: The model name (e.g., "gpt-4o")
            config: Rate limit configuration
        """
        if provider not in self._provider_model_configs:
            self._provider_model_configs[provider] = {}
            self._provider_model_states[provider] = {}
            self._locks[provider] = {}

        self._provider_model_configs[provider][model] = config

        # Initialize state
        self._provider_model_states[provider][model] = {
            "request_timestamps": [],
            "token_usage": [],
            "last_cleanup_time": time.time(),
        }

        # Initialize lock
        self._locks[provider][model] = threading.Lock()

        # Start monitoring if enabled and not already started
        self._ensure_monitor_started()
        logger.info(f"Registered {provider}/{model}")

    def register_config(self, config: RateLimitConfig) -> None:
        """Register a rate limit configuration.

        Args:
            config: Rate limit configuration
        """
        self._config = config
        for provider, provider_config in config.providers.items():
            for model, model_config in provider_config.models.items():
                self.register_model(provider, model, model_config)

    def load_config(self, config_path: str) -> None:
        """Load a rate limit configuration from a file.

        Args:
            config_path: Path to the rate limit configuration file
        """
        if config_path == "default":
            config_path = os.path.join(os.path.dirname(__file__), DEFAULT_RATE_LIMIT_CONFIG_PATH)
        self.register_config(load_rate_limit_configs(config_path))

    def get_config(self, provider: str, model: str) -> Optional[ModelRateLimitConfig]:
        """Get the rate limit configuration for a provider/model.

        Args:
            provider: The provider name
            model: The model name

        Returns:
            The rate limit configuration if it exists, None otherwise
        """
        if provider not in self._provider_model_configs:
            return None
        return self._provider_model_configs[provider].get(model)

    def _cleanup_old_data(self, provider: str, model: str) -> None:
        """Clean up old timestamps and token usage data.

        Args:
            provider: The provider name
            model: The model name
        """
        state = self._provider_model_states[provider][model]
        current_time = time.time()

        # Clean up only once every few seconds to avoid excessive cleaning
        if current_time - state["last_cleanup_time"] < 5:
            return

        one_minute_ago = current_time - 60

        # Filter out timestamps older than one minute
        state["request_timestamps"] = [
            ts for ts in state["request_timestamps"] if ts > one_minute_ago
        ]

        # Filter out token usage older than one minute
        state["token_usage"] = [
            (ts, tokens) for ts, tokens in state["token_usage"] if ts > one_minute_ago
        ]

        state["last_cleanup_time"] = current_time

    async def acquire(self, provider: str, model: str, tokens: int = 0) -> bool:
        """Acquire permission to make an API call.

        Args:
            provider: The provider name
            model: The model name
            tokens: Estimated token usage for this request

        Returns:
            True if permission is granted, False otherwise
        """
        config = self.get_config(provider, model)
        if not config:
            logger.warning(f"No rate limit config for {provider}/{model}, allowing request")
            return True

        with self._locks[provider][model]:
            self._cleanup_old_data(provider, model)
            state = self._provider_model_states[provider][model]
            current_time = time.time()

            # Check requests per minute
            if config.requests_per_minute > 0:
                current_rpm = len(state["request_timestamps"])
                if current_rpm >= config.requests_per_minute:
                    return False

            # Check tokens per minute
            if config.tokens_per_minute > 0 and tokens > 0:
                current_tpm = sum(tok for _, tok in state["token_usage"])
                if current_tpm + tokens > config.tokens_per_minute:
                    return False

            # Update state with this request
            state["request_timestamps"].append(current_time)
            if tokens > 0:
                state["token_usage"].append((current_time, tokens))

            return True

    async def wait_and_acquire(self, provider: str, model: str, tokens: int = 0) -> bool:
        """Wait until rate limit allows and acquire permission.

        Args:
            provider: The provider name
            model: The model name
            tokens: Estimated token usage for this request

        Returns:
            True when permission is granted

        Raises:
            RuntimeError: If unable to acquire permission within 10 minutes
        """
        start_time = time.time()
        timeout = 600  # 10 minutes in seconds

        while True:
            if await self.acquire(provider, model, tokens):
                return True

            # Check if we've exceeded the timeout
            if time.time() - start_time > timeout:
                raise RuntimeError(
                    f"Failed to acquire rate limit for {provider}/{model} after 10 minutes"
                )
            await asyncio.sleep(1)

    def record_usage(self, provider: str, model: str, tokens_used: int) -> None:
        """Record actual token usage after a request.

        This is used to update token usage tracking with actual values rather than estimates.

        Args:
            provider: The provider name
            model: The model name
            tokens_used: Actual tokens used in the request
        """
        if not self.get_config(provider, model):
            return

        with self._locks[provider][model]:
            state = self._provider_model_states[provider][model]

            # Update the most recent token usage record
            if state["token_usage"]:
                timestamp, _ = state["token_usage"][-1]
                state["token_usage"][-1] = (timestamp, tokens_used)

    def get_usage_stats(
        self, provider: Optional[str] = None, model: Optional[str] = None
    ) -> dict[str, Any]:
        """Get current usage statistics.

        Args:
            provider: The provider name (optional, if None returns stats for all providers)
            model: The model name (optional, if None returns stats for all models of the provider)

        Returns:
            Dictionary with current usage statistics
        """
        # Start monitoring if enabled and not already started
        self._ensure_monitor_started()

        result: dict[str, Any] = {}

        if provider is None:
            # Return stats for all providers
            for provider_name in self.providers:
                result[provider_name] = self.get_usage_stats(provider_name)
            return result

        if provider not in self._provider_model_configs:
            return {"error": "Provider not registered"}

        if model is None:
            # Return stats for all models of this provider
            provider_stats = {}
            for model_name in self._provider_model_configs[provider]:
                provider_stats[model_name] = self.get_usage_stats(provider, model_name)
            return provider_stats

        if model not in self._provider_model_configs[provider]:
            return {"error": "Model not registered"}

        with self._locks[provider][model]:
            self._cleanup_old_data(provider, model)
            state = self._provider_model_states[provider][model]

            config = self._provider_model_configs[provider][model]
            current_rpm = len(state["request_timestamps"])
            current_tpm = sum(tok for _, tok in state["token_usage"])

            return {
                "requests_per_minute": {
                    "current": current_rpm,
                    "limit": config.requests_per_minute,
                    "percent": (
                        (current_rpm / config.requests_per_minute * 100)
                        if config.requests_per_minute
                        else 0
                    ),
                },
                "tokens_per_minute": {
                    "current": current_tpm,
                    "limit": config.tokens_per_minute,
                    "percent": (
                        (current_tpm / config.tokens_per_minute * 100)
                        if config.tokens_per_minute
                        else 0
                    ),
                },
            }


# Global rate limiter instance
rate_limiter = RateLimit()

# Load rate limit config from file
if "RATE_LIMIT_CONFIG_PATH" not in os.environ:
    logger.warning("RATE_LIMIT_CONFIG_PATH not set, using default `Free / Tier 1` config")
    rate_limiter.load_config(
        os.path.join(os.path.dirname(__file__), DEFAULT_RATE_LIMIT_CONFIG_PATH)
    )
else:
    rate_limiter.load_config(os.environ["RATE_LIMIT_CONFIG_PATH"])
