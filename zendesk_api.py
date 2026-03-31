"""Zendesk API client — fetches tickets, posts private notes."""
import logging
import os

import httpx

logger = logging.getLogger(__name__)

ZENDESK_SUBDOMAIN = os.getenv("ZENDESK_SUBDOMAIN")  # e.g. "dagoexpress"
ZENDESK_EMAIL = os.getenv("ZENDESK_EMAIL")  # e.g. "agent@dagoexpress.de"
ZENDESK_API_TOKEN = os.getenv("ZENDESK_API_TOKEN")
CARRIER_SUPPORT_GROUP_NAME = os.getenv("CARRIER_SUPPORT_GROUP", "Carrier Support")

_carrier_support_group_id: int | None = None


def _base_url() -> str:
    return f"https://{ZENDESK_SUBDOMAIN}.zendesk.com/api/v2"


def _auth() -> tuple[str, str]:
    return (f"{ZENDESK_EMAIL}/token", ZENDESK_API_TOKEN)


async def resolve_carrier_support_group_id() -> int | None:
    """Find the group ID for Carrier Support (cached after first call)."""
    global _carrier_support_group_id
    if _carrier_support_group_id is not None:
        return _carrier_support_group_id

    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{_base_url()}/groups.json",
            auth=_auth(),
            timeout=10.0,
        )
        if resp.status_code != 200:
            logger.error(f"Failed to fetch groups: {resp.status_code} {resp.text[:200]}")
            return None

        for group in resp.json().get("groups", []):
            if group["name"] == CARRIER_SUPPORT_GROUP_NAME:
                _carrier_support_group_id = group["id"]
                logger.info(f"Carrier Support group ID: {_carrier_support_group_id}")
                return _carrier_support_group_id

    logger.warning(f"Group '{CARRIER_SUPPORT_GROUP_NAME}' not found in Zendesk")
    return None


async def get_ticket(ticket_id: int) -> dict | None:
    """Fetch a single ticket by ID."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{_base_url()}/tickets/{ticket_id}.json",
            auth=_auth(),
            timeout=10.0,
        )
        if resp.status_code != 200:
            logger.error(f"Failed to fetch ticket {ticket_id}: {resp.status_code}")
            return None
        return resp.json().get("ticket")


async def get_ticket_comments(ticket_id: int) -> list[dict]:
    """Fetch all comments on a ticket."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{_base_url()}/tickets/{ticket_id}/comments.json",
            auth=_auth(),
            timeout=10.0,
        )
        if resp.status_code != 200:
            logger.error(f"Failed to fetch comments for ticket {ticket_id}: {resp.status_code}")
            return []
        return resp.json().get("comments", [])


async def get_requester(user_id: int) -> dict | None:
    """Fetch user info."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{_base_url()}/users/{user_id}.json",
            auth=_auth(),
            timeout=10.0,
        )
        if resp.status_code != 200:
            return None
        return resp.json().get("user")


async def post_private_note(ticket_id: int, body: str) -> bool:
    """Post an internal (private) note on a ticket."""
    payload = {
        "ticket": {
            "comment": {
                "body": body,
                "public": False,
            }
        }
    }
    async with httpx.AsyncClient() as client:
        resp = await client.put(
            f"{_base_url()}/tickets/{ticket_id}.json",
            auth=_auth(),
            json=payload,
            timeout=15.0,
        )
        if resp.status_code != 200:
            logger.error(f"Failed to post note on ticket {ticket_id}: {resp.status_code} {resp.text[:200]}")
            return False
        logger.info(f"Private note posted on ticket {ticket_id}")
        return True
