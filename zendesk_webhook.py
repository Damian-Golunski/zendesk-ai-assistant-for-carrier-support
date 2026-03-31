"""Zendesk webhook handler — receives ticket events, triggers AI analysis."""
import logging
import os

from fastapi import APIRouter, Request, HTTPException

from zendesk_api import (
    get_ticket,
    get_ticket_comments,
    get_requester,
    post_private_note,
    resolve_carrier_support_group_id,
)
from ai_assistant import analyze_ticket

logger = logging.getLogger(__name__)

router = APIRouter()

WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "")


@router.post("/webhook/zendesk")
async def handle_zendesk_webhook(request: Request):
    """Handle Zendesk webhook for new/updated tickets.

    Zendesk sends a POST when a ticket is created or updated.
    We only process tickets assigned to the Carrier Support group.
    We only add a note on the FIRST message (ticket creation).
    """
    # Verify webhook secret if configured
    if WEBHOOK_SECRET:
        token = request.headers.get("X-Webhook-Secret", "")
        if token != WEBHOOK_SECRET:
            raise HTTPException(status_code=401, detail="Invalid webhook secret")

    body = await request.json()
    logger.info(f"Webhook received: {body}")

    ticket_id = body.get("ticket_id")
    if not ticket_id:
        # Try alternative payload formats
        ticket_id = body.get("id") or (body.get("ticket", {}) or {}).get("id")

    if not ticket_id:
        logger.warning("No ticket_id in webhook payload")
        return {"status": "skipped", "reason": "no ticket_id"}

    ticket_id = int(ticket_id)

    # Fetch the ticket
    ticket = await get_ticket(ticket_id)
    if not ticket:
        return {"status": "error", "reason": "ticket not found"}

    # Check if ticket belongs to Carrier Support group
    carrier_group_id = await resolve_carrier_support_group_id()
    ticket_group_id = ticket.get("group_id")

    if carrier_group_id and ticket_group_id != carrier_group_id:
        logger.info(f"Ticket {ticket_id} not in Carrier Support group (group_id={ticket_group_id}), skipping")
        return {"status": "skipped", "reason": "not carrier support group"}

    # Get comments to check if this is the first message
    comments = await get_ticket_comments(ticket_id)

    # Filter out system/internal comments — only count public end-user comments
    public_comments = [c for c in comments if c.get("public", True) and c.get("author_id") == ticket.get("requester_id")]

    if len(public_comments) != 1:
        logger.info(f"Ticket {ticket_id} has {len(public_comments)} requester comments, skipping (only process first)")
        return {"status": "skipped", "reason": "not first message"}

    first_comment = public_comments[0]
    message_body = first_comment.get("plain_body") or first_comment.get("body", "")
    subject = ticket.get("subject", "(kein Betreff)")

    # Get requester info
    requester_name = None
    requester_id = ticket.get("requester_id")
    if requester_id:
        requester = await get_requester(requester_id)
        if requester:
            requester_name = requester.get("name")

    # Run AI analysis
    try:
        analysis = await analyze_ticket(subject, message_body, requester_name)
    except Exception as e:
        logger.error(f"AI analysis failed for ticket {ticket_id}: {e}")
        return {"status": "error", "reason": "ai analysis failed"}

    # Post as private note
    note_body = f"🤖 AI Carrier Support Assistant\n\n{analysis}"
    success = await post_private_note(ticket_id, note_body)

    return {"status": "ok" if success else "error", "ticket_id": ticket_id}
