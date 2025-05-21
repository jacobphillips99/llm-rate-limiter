"""
Simple rate limiter for LLM API calls.
"""

import logging
import os

# Get the logger for this package
logger = logging.getLogger(__name__)

log_level_name = os.environ.get("LLM_RATE_LIMIT_LOG_LEVEL", "ERROR").upper()
log_level = getattr(logging, log_level_name, logging.INFO)  # fallback to INFO if invalid level

# Configure the logger for this package
logger.setLevel(log_level)

# Add a handler if no handlers are configured for this logger or its ancestors
if not logger.handlers and not logging.getLogger().handlers:  # Check root logger too
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    # Prevent messages from being passed to the root logger's handlers
    # if the root logger is also configured, to avoid duplicate messages.
    # This is generally good practice unless specific propagation is desired.
    logger.propagate = False
