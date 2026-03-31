"""Zendesk Carrier Support AI Assistant for DAGO Express."""
import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

from zendesk_webhook import router as webhook_router

app = FastAPI(title="Zendesk Carrier Support AI Assistant")
app.include_router(webhook_router)


@app.get("/")
async def health():
    return {"status": "ok", "service": "zendesk-carrier-ai"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
