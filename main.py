"""Zendesk Carrier Support AI Assistant for DAGO Express."""
import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Validate config before importing anything else (punkt 3)
from config import validate_config
validate_config()

# Sentry for error alerting (punkt 7)
import sentry_sdk
sentry_dsn = os.getenv("SENTRY_DSN")
if sentry_dsn:
    sentry_sdk.init(dsn=sentry_dsn, traces_sample_rate=0.1)
    logging.getLogger(__name__).info("Sentry initialized for error tracking")

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from zendesk_api import close_client
from zendesk_webhook import router as webhook_router, limiter


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Connect to DAGO Hub
    import hub_client
    await hub_client.start()
    yield
    # Cleanup
    await hub_client.stop()
    await close_client()


app = FastAPI(title="Zendesk Carrier Support AI Assistant", debug=False, lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.include_router(webhook_router)


@app.get("/")
async def health():
    return {"status": "ok", "service": "zendesk-carrier-ai"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
