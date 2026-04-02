"""Zendesk webhook handler — receives ticket events, triggers AI analysis."""
import logging
import os

from fastapi import APIRouter, Request, HTTPException

from zendesk_api import (
    get_ticket,
    get_ticket_comments,
    get_requester,
    post_private_note,
    post_public_reply,
    resolve_carrier_support_group_id,
)
from ai_assistant import analyze_ticket, analyze_follow_up, generate_bewerbung_reply

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

    # Get comments
    comments = await get_ticket_comments(ticket_id)
    subject = ticket.get("subject", "(kein Betreff)")

    # Get requester info
    requester_name = None
    requester_id = ticket.get("requester_id")
    if requester_id:
        requester = await get_requester(requester_id)
        if requester:
            requester_name = requester.get("name")

    # Filter public end-user comments
    public_comments = [c for c in comments if c.get("public", True) and c.get("author_id") == requester_id]

    if not public_comments:
        return {"status": "skipped", "reason": "no requester comments"}

    # Get first message body (needed for both analysis and Bewerbung detection)
    first_comment = public_comments[0]
    message_body = first_comment.get("plain_body") or first_comment.get("body", "")

    is_follow_up = len(public_comments) > 1

    if is_follow_up:
        # Build conversation history from all comments
        conversation = []
        for c in comments:
            is_requester = c.get("author_id") == requester_id
            is_public = c.get("public", True)
            text = c.get("plain_body") or c.get("body", "")
            if is_requester and is_public:
                conversation.append({"role": "Carrier", "text": text})
            elif not is_requester and is_public:
                conversation.append({"role": "Agent (öffentlich)", "text": text})
            elif not is_public:
                conversation.append({"role": "Interne Notiz", "text": text})

        try:
            analysis = await analyze_follow_up(subject, conversation, requester_name)
        except Exception as e:
            logger.error(f"AI follow-up analysis failed for ticket {ticket_id}: {e}")
            return {"status": "error", "reason": "ai analysis failed"}

        note_body = f"🤖 AI Carrier Support Assistant (Follow-up)\n\n{analysis}"
        success = await post_private_note(ticket_id, note_body)
    else:
        # Run AI analysis
        try:
            analysis = await analyze_ticket(subject, message_body, requester_name)
        except Exception as e:
            logger.error(f"AI analysis failed for ticket {ticket_id}: {e}")
            return {"status": "error", "reason": "ai analysis failed"}

        # Post as private note
        note_body = f"🤖 AI Carrier Support Assistant\n\n{analysis}"
        success = await post_private_note(ticket_id, note_body)

    # Auto-reply for Bewerbung/CV tickets (only on first message)
    if not is_follow_up:
        bewerbung_keywords = [
            # DE
            "bewerbung", "bewerben", "lebenslauf", "stellenangebot",
            # EN
            "cv ", "curriculum vitae", "resume", "job application", "apply for", "job offer",
            # ES
            "candidatura", "solicitud de empleo", "oferta de trabajo", "currículum",
            # FR
            "candidature", "offre d'emploi", "lettre de motivation",
            # IT
            "candidatura", "offerta di lavoro", "domanda di lavoro",
            # NL
            "sollicitatie", "vacature", "curriculum vitae",
            # PL
            "podanie o prace", "aplikacja", "praca kierowca", "oferta pracy",
            # HU
            "állásjelentkezés", "önéletrajz", "álláspályázat",
            # CS
            "žádost o práci", "životopis", "nabídka práce",
            # SK
            "žiadosť o prácu", "životopis", "ponuka práce",
            # RO
            "cerere de angajare", "angajare", "locuri de muncă",
            # NO/SV/DA/FI
            "jobbsøknad", "jobbansökan", "jobansøgning", "työhakemus",
            # SI/HR
            "prijava za delo", "prijava za posao", "životopis",
            # LT/ET/LV
            "darbo paraiška", "tööavaldus", "darba pieteikums",
            # EL
            "αίτηση εργασίας", "βιογραφικό",
            # BG
            "кандидатура", "автобиография", "обява за работа",
        ]
        text_lower = f"{subject} {message_body}".lower()
        is_bewerbung = any(kw in text_lower for kw in bewerbung_keywords)

        if is_bewerbung:
            logger.info(f"Ticket {ticket_id} detected as Bewerbung — generating auto-reply in sender's language")
            try:
                auto_reply = await generate_bewerbung_reply(subject, message_body)
                await post_public_reply(ticket_id, auto_reply, status="solved")
            except Exception as e:
                logger.error(f"Failed to generate/send Bewerbung auto-reply for ticket {ticket_id}: {e}")
                is_bewerbung = False
    else:
        is_bewerbung = False

    return {"status": "ok" if success else "error", "ticket_id": ticket_id, "auto_replied": is_bewerbung}
