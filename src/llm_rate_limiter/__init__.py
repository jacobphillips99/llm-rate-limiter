"""
Simple rate limiter for LLM API calls.
"""

import logging
import os

log_level = os.environ.get("LLM_RATE_LIMIT_LOG_LEVEL", "ERROR").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),  # fallback to INFO if invalid level
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
