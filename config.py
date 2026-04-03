"""Startup configuration validation — fail-fast if required env vars are missing."""
import os
import sys
import logging

logger = logging.getLogger(__name__)

REQUIRED_ENV_VARS = [
    "ZENDESK_SUBDOMAIN",
    "ZENDESK_EMAIL",
    "ZENDESK_API_TOKEN",
    "ANTHROPIC_API_KEY",
    "WEBHOOK_SECRET",
]


def validate_config():
    """Validate all required environment variables are set. Exit if not."""
    missing = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]
    if missing:
        logger.critical(f"Missing required environment variables: {', '.join(missing)}")
        logger.critical("Refusing to start — set all required env vars and restart.")
        sys.exit(1)

    # Warn if API_KEY is optional but recommended
    api_key = os.getenv("API_KEY")
    if not api_key:
        logger.warning("API_KEY not set — admin endpoints (/ticket/*/note, /reply, /comments) will be unprotected")
