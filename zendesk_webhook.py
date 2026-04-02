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


@router.get("/tickets/carrier-support/recent")
async def list_recent_carrier_support():
    """List all open/new Carrier Support tickets (paginated)."""
    import httpx
    from zendesk_api import _base_url, _auth
    carrier_group_id = await resolve_carrier_support_group_id()
    all_results = []
    async with httpx.AsyncClient() as client:
        page = 1
        while page <= 5:
            resp = await client.get(
                f"{_base_url()}/search.json",
                params={
                    "query": f"type:ticket group_id:{carrier_group_id} status<solved order_by:created_at sort:desc",
                    "page": page,
                    "per_page": 100,
                },
                auth=_auth(),
                timeout=15.0,
            )
            data = resp.json()
            all_results.extend(data.get("results", []))
            if not data.get("next_page"):
                break
            page += 1
    tickets = []
    for t in all_results:
        tickets.append({
            "id": t["id"],
            "subject": t.get("subject", ""),
            "status": t.get("status", ""),
            "created_at": t.get("created_at", ""),
        })
    return {"count": len(tickets), "tickets": tickets}


@router.get("/ticket/{ticket_id}/comments")
async def get_comments(ticket_id: int):
    """Read all comments for a ticket (debug/review endpoint)."""
    comments = await get_ticket_comments(ticket_id)
    ticket = await get_ticket(ticket_id)
    subject = ticket.get("subject", "") if ticket else ""
    result = []
    for c in comments:
        result.append({
            "author_id": c.get("author_id"),
            "public": c.get("public", True),
            "body": c.get("plain_body") or c.get("body", ""),
            "created_at": c.get("created_at"),
        })
    return {"ticket_id": ticket_id, "subject": subject, "comments": result}


@router.post("/ticket/{ticket_id}/reply")
async def send_reply(ticket_id: int, request: Request):
    """Send a public reply on a ticket and optionally set status."""
    body = await request.json()
    text = body.get("body", "")
    status = body.get("status", "solved")
    if not text:
        raise HTTPException(status_code=400, detail="body is required")
    success = await post_public_reply(ticket_id, text, status=status)
    return {"status": "ok" if success else "error", "ticket_id": ticket_id}


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
            # Skip previous AI notes to avoid polluting context
            if not is_public and "🤖 AI Carrier Support" in text:
                continue
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

        # Check if AI flagged this follow-up for auto-reply
        if "AUTO-REPLY: JA" in analysis:
            lines = analysis.split("\n")
            reply_text = ""
            for i, line in enumerate(lines):
                if "ANTWORTVORSCHLAG" in line:
                    reply_text = "\n".join(lines[i + 1:]).strip()
                    break
            if reply_text:
                logger.info(f"Ticket {ticket_id} follow-up auto-reply (AI flagged AUTO-REPLY: JA)")
                try:
                    await post_public_reply(ticket_id, reply_text, status="solved")
                    return {"status": "ok", "ticket_id": ticket_id, "auto_replied": True}
                except Exception as e:
                    logger.error(f"Failed to send follow-up auto-reply for ticket {ticket_id}: {e}")

        # No auto-reply — post as private note for agent
        note_body = f"🤖 AI Carrier Support Assistant (Follow-up)\n\n{analysis}"
        success = await post_private_note(ticket_id, note_body)
    else:
        # Run AI analysis
        try:
            analysis = await analyze_ticket(subject, message_body, requester_name)
        except Exception as e:
            logger.error(f"AI analysis failed for ticket {ticket_id}: {e}")
            return {"status": "error", "reason": "ai analysis failed"}

        # Auto-reply for specific categories (only on first message)
        # Extract category from AI analysis (line after "KATEGORIE" header)
        category = ""
        lines = analysis.split("\n")
        for i, line in enumerate(lines):
            if "KATEGORIE" in line:
                for j in range(i + 1, min(i + 3, len(lines))):
                    candidate = lines[j].strip()
                    if candidate:
                        category = candidate.lower()
                        break
                break

        # Categories eligible for auto-reply using the Antwortvorschlag from analysis
        auto_reply_categories = {"versicherung", "registrierung", "auftraege"}
        # Normalize category: replace umlauts and encoding variants
        import unicodedata
        category = unicodedata.normalize("NFC", category)
        category = category.replace("ä", "ae").replace("ü", "ue").replace("ö", "oe")

        # Check for Bewerbung via keywords (more reliable than AI category for CVs)
        bewerbung_keywords = [
            "bewerbung", "bewerben", "lebenslauf", "stellenangebot",
            "cv ", "curriculum vitae", "resume", "job application", "apply for", "job offer",
            "candidatura", "solicitud de empleo", "oferta de trabajo", "currículum",
            "candidature", "offre d'emploi", "lettre de motivation",
            "offerta di lavoro", "domanda di lavoro",
            "sollicitatie", "vacature",
            "podanie o prace", "aplikacja", "praca kierowca", "oferta pracy",
            "állásjelentkezés", "önéletrajz", "álláspályázat",
            "žádost o práci", "životopis", "nabídka práce",
            "žiadosť o prácu", "ponuka práce",
            "cerere de angajare", "angajare", "locuri de muncă",
            "jobbsøknad", "jobbansökan", "jobansøgning", "työhakemus",
            "prijava za delo", "prijava za posao",
            "darbo paraiška", "tööavaldus", "darba pieteikums",
            "αίτηση εργασίας", "βιογραφικό",
            "кандидатура", "автобиография", "обява за работа",
        ]
        text_lower = f"{subject} {message_body}".lower()
        is_bewerbung = any(kw in text_lower for kw in bewerbung_keywords)

        if is_bewerbung:
            logger.info(f"Ticket {ticket_id} detected as Bewerbung — auto-reply")
            try:
                auto_reply = await generate_bewerbung_reply(subject, message_body)
                await post_public_reply(ticket_id, auto_reply, status="solved")
                return {"status": "ok", "ticket_id": ticket_id, "auto_replied": True}
            except Exception as e:
                logger.error(f"Failed to send Bewerbung auto-reply for ticket {ticket_id}: {e}")

        elif category in auto_reply_categories:
            reply_text = ""
            for i, line in enumerate(lines):
                if "ANTWORTVORSCHLAG" in line:
                    reply_text = "\n".join(lines[i + 1:]).strip()
                    break
            if reply_text:
                logger.info(f"Ticket {ticket_id} category '{category}' — auto-reply with Antwortvorschlag")
                try:
                    await post_public_reply(ticket_id, reply_text, status="solved")
                    return {"status": "ok", "ticket_id": ticket_id, "auto_replied": True}
                except Exception as e:
                    logger.error(f"Failed to send auto-reply for ticket {ticket_id}: {e}")

        # No auto-reply — post as private note for agent
        note_body = f"🤖 AI Carrier Support Assistant\n\n{analysis}"
        success = await post_private_note(ticket_id, note_body)

    return {"status": "ok" if success else "error", "ticket_id": ticket_id, "auto_replied": False}
