"""Zendesk API client — fetches tickets, posts private notes."""
import logging
import os

import httpx

logger = logging.getLogger(__name__)

ZENDESK_SUBDOMAIN = os.getenv("ZENDESK_SUBDOMAIN")
ZENDESK_EMAIL = os.getenv("ZENDESK_EMAIL")
ZENDESK_API_TOKEN = os.getenv("ZENDESK_API_TOKEN")
CARRIER_SUPPORT_GROUP_NAME = os.getenv("CARRIER_SUPPORT_GROUP", "Carrier Support")

_carrier_support_group_id: int | None = None

# Global httpx client for connection pooling (punkt 12)
_http_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(timeout=15.0)
    return _http_client


async def close_client():
    global _http_client
    if _http_client and not _http_client.is_closed:
        await _http_client.aclose()
        _http_client = None


def _base_url() -> str:
    return f"https://{ZENDESK_SUBDOMAIN}.zendesk.com/api/v2"


def _auth() -> tuple[str, str]:
    return (f"{ZENDESK_EMAIL}/token", ZENDESK_API_TOKEN)


async def resolve_carrier_support_group_id() -> int | None:
    """Find the group ID for Carrier Support (cached after first call)."""
    global _carrier_support_group_id
    if _carrier_support_group_id is not None:
        return _carrier_support_group_id

    client = _get_client()
    resp = await client.get(
        f"{_base_url()}/groups.json",
        auth=_auth(),
        timeout=10.0,
    )
    if resp.status_code != 200:
        logger.error(f"Failed to fetch groups: {resp.status_code}")
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
    client = _get_client()
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
    client = _get_client()
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
    client = _get_client()
    resp = await client.get(
        f"{_base_url()}/users/{user_id}.json",
        auth=_auth(),
        timeout=10.0,
    )
    if resp.status_code != 200:
        return None
    return resp.json().get("user")


async def add_tag(ticket_id: int, tag: str) -> bool:
    """Add a tag to a ticket (used for idempotency)."""
    client = _get_client()
    resp = await client.put(
        f"{_base_url()}/tickets/{ticket_id}/tags.json",
        auth=_auth(),
        json={"tags": [tag]},
        timeout=10.0,
    )
    if resp.status_code != 200:
        logger.error(f"Failed to add tag '{tag}' to ticket {ticket_id}: {resp.status_code}")
        return False
    return True


async def post_private_note(ticket_id: int, body: str) -> bool:
    """Post an internal (private) note on a ticket."""
    payload = {
        "ticket": {
            "comment": {
                "html_body": body.replace("\n", "<br>"),
                "public": False,
            }
        }
    }
    client = _get_client()
    resp = await client.put(
        f"{_base_url()}/tickets/{ticket_id}.json",
        auth=_auth(),
        json=payload,
        timeout=15.0,
    )
    if resp.status_code != 200:
        logger.error(f"Failed to post note on ticket {ticket_id}: {resp.status_code}")
        return False
    logger.info(f"Private note posted on ticket {ticket_id}")
    return True


async def post_public_reply(ticket_id: int, body: str, status: str = "solved") -> bool:
    """Post a public reply on a ticket and optionally set status."""
    payload = {
        "ticket": {
            "comment": {
                "body": body,
                "public": True,
            },
            "status": status,
        }
    }
    client = _get_client()
    resp = await client.put(
        f"{_base_url()}/tickets/{ticket_id}.json",
        auth=_auth(),
        json=payload,
        timeout=15.0,
    )
    if resp.status_code != 200:
        logger.error(f"Failed to post reply on ticket {ticket_id}: {resp.status_code}")
        return False
    logger.info(f"Public reply posted on ticket {ticket_id} (status={status})")
    return True
