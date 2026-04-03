"""Zendesk webhook handler — receives ticket events, triggers AI analysis."""
import asyncio
import hmac
import logging
import os

from fastapi import APIRouter, Request, HTTPException, Depends, Security
from fastapi.security import APIKeyHeader
from slowapi import Limiter
from slowapi.util import get_remote_address

from zendesk_api import (
    get_ticket,
    get_ticket_comments,
    get_requester,
    post_private_note,
    post_public_reply,
    add_tag,
    resolve_carrier_support_group_id,
)
from ai_assistant import analyze_ticket, analyze_follow_up, generate_bewerbung_reply

logger = logging.getLogger(__name__)

router = APIRouter()

# --- Punkt 4: Webhook secret required (no default) ---
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET")

# --- Punkt 5: API key auth for admin endpoints ---
API_KEY = os.getenv("API_KEY", "")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Security(api_key_header)):
    """Require API key for admin endpoints."""
    if not API_KEY:
        return  # No API_KEY configured — skip (warned at startup)
    if not api_key or not hmac.compare_digest(api_key, API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# --- Punkt 8: Rate limiting ---
limiter = Limiter(key_func=get_remote_address)

# --- Punkt 2: Idempotency tag ---
AI_PROCESSED_TAG = "ai_processed"


@router.get("/tickets/carrier-support/recent", dependencies=[Depends(verify_api_key)])
@limiter.limit("10/minute")
async def list_recent_carrier_support(request: Request):
    """List all open/new Carrier Support tickets (paginated)."""
    from zendesk_api import _get_client, _base_url, _auth
    carrier_group_id = await resolve_carrier_support_group_id()
    all_results = []
    client = _get_client()
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


async def _verify_carrier_support(ticket_id: int):
    """Verify ticket belongs to Carrier Support group. Raises 403 if not."""
    carrier_group_id = await resolve_carrier_support_group_id()
    if not carrier_group_id:
        raise HTTPException(status_code=503, detail="Could not resolve Carrier Support group")
    ticket = await get_ticket(ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")
    if ticket.get("group_id") != carrier_group_id:
        raise HTTPException(status_code=403, detail="Ticket does not belong to Carrier Support group")
    return ticket


@router.get("/ticket/{ticket_id}/comments", dependencies=[Depends(verify_api_key)])
@limiter.limit("10/minute")
async def get_comments(ticket_id: int, request: Request):
    """Read all comments for a ticket (debug/review endpoint)."""
    ticket = await _verify_carrier_support(ticket_id)
    comments = await get_ticket_comments(ticket_id)
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


@router.post("/ticket/{ticket_id}/note", dependencies=[Depends(verify_api_key)])
@limiter.limit("10/minute")
async def send_note(ticket_id: int, request: Request):
    """Post a private note on a ticket."""
    await _verify_carrier_support(ticket_id)
    body = await request.json()
    text = body.get("body", "")
    if not text:
        raise HTTPException(status_code=400, detail="body is required")
    success = await post_private_note(ticket_id, text)
    return {"status": "ok" if success else "error", "ticket_id": ticket_id}


@router.post("/ticket/{ticket_id}/reply", dependencies=[Depends(verify_api_key)])
@limiter.limit("10/minute")
async def send_reply(ticket_id: int, request: Request):
    """Send a public reply on a ticket and optionally set status."""
    await _verify_carrier_support(ticket_id)
    body = await request.json()
    text = body.get("body", "")
    status = body.get("status", "solved")
    if not text:
        raise HTTPException(status_code=400, detail="body is required")
    success = await post_public_reply(ticket_id, text, status=status)
    return {"status": "ok" if success else "error", "ticket_id": ticket_id}


# --- Punkt 1: Output validation for auto-reply safety ---
def _validate_auto_reply(analysis: str, reply_text: str) -> bool:
    """Validate that auto-reply output is safe and not prompt-injected."""
    if not reply_text or len(reply_text) < 10:
        return False
    if len(reply_text) > 3000:
        logger.warning("Auto-reply too long, likely injection")
        return False
    # Check for suspicious patterns that indicate prompt injection
    suspicious_patterns = [
        "ignore previous", "ignore all", "system prompt", "instructions above",
        "DAGO Express has been hacked", "account deleted", "password",
        "credit card", "bank account", "wire transfer",
    ]
    reply_lower = reply_text.lower()
    for pattern in suspicious_patterns:
        if pattern in reply_lower:
            logger.warning(f"Auto-reply blocked: suspicious pattern '{pattern}' detected")
            return False
    return True


@router.post("/webhook/zendesk")
@limiter.limit("30/minute")
async def handle_zendesk_webhook(request: Request):
    """Handle Zendesk webhook for new/updated tickets.

    Zendesk sends a POST when a ticket is created or updated.
    We only process tickets assigned to the Carrier Support group.
    """
    # --- Punkt 4: Webhook secret REQUIRED, timing-safe comparison ---
    if not WEBHOOK_SECRET:
        logger.error("WEBHOOK_SECRET not configured — rejecting all webhooks")
        raise HTTPException(status_code=503, detail="Webhook not configured")

    token = request.headers.get("X-Webhook-Secret", "")
    if not hmac.compare_digest(token, WEBHOOK_SECRET):
        raise HTTPException(status_code=401, detail="Unauthorized")

    body = await request.json()

    # --- Punkt 14: Only log ticket_id and group, not full payload (PII) ---
    ticket_id = body.get("ticket_id")
    if not ticket_id:
        ticket_id = body.get("id") or (body.get("ticket", {}) or {}).get("id")

    if not ticket_id:
        logger.warning("No ticket_id in webhook payload")
        return {"status": "skipped", "reason": "no ticket_id"}

    ticket_id = int(ticket_id)
    logger.info(f"Webhook received for ticket {ticket_id}")

    # Fetch the ticket
    ticket = await get_ticket(ticket_id)
    if not ticket:
        return {"status": "error", "reason": "ticket not found"}

    # --- Punkt 2: Idempotency check ---
    # Use comment count in tag to allow follow-up processing but prevent duplicates
    existing_tags = ticket.get("tags", [])
    comment_count = len([c for c in (await get_ticket_comments(ticket_id)) if c.get("public", True)])
    current_marker = f"ai_processed_{comment_count}"
    if current_marker in existing_tags:
        logger.info(f"Ticket {ticket_id} already processed for {comment_count} comments, skipping")
        return {"status": "skipped", "reason": "already processed"}

    # Check if ticket belongs to Carrier Support group (FAIL-CLOSED)
    carrier_group_id = await resolve_carrier_support_group_id()
    if not carrier_group_id:
        logger.error("Could not resolve Carrier Support group ID — refusing to process ticket")
        return {"status": "error", "reason": "carrier support group not resolved"}

    ticket_group_id = ticket.get("group_id")
    if ticket_group_id != carrier_group_id:
        logger.info(f"Ticket {ticket_id} not in Carrier Support group (group_id={ticket_group_id}), skipping")
        return {"status": "skipped", "reason": "not carrier support group"}

    # --- Punkt 6: Overall timeout for the entire processing ---
    try:
        result = await asyncio.wait_for(
            _process_ticket(ticket_id, ticket, current_marker),
            timeout=60.0,
        )
        return result
    except asyncio.TimeoutError:
        logger.error(f"Ticket {ticket_id} processing timed out after 60s")
        return {"status": "error", "reason": "processing timeout"}


async def _process_ticket(ticket_id: int, ticket: dict, idempotency_marker: str) -> dict:
    """Process a single ticket — extracted for timeout wrapping."""

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

    # Get first message body
    first_comment = public_comments[0]
    message_body = first_comment.get("plain_body") or first_comment.get("body", "")

    is_follow_up = len(public_comments) > 1
    success = False

    if is_follow_up:
        # Quick-close: if last message is ONLY a thank-you (short, no real content)
        last_comment = public_comments[-1]
        last_text = (last_comment.get("plain_body") or last_comment.get("body", "")).strip()
        last_text_lower = last_text.lower()
        last_text_clean = last_text_lower.rstrip("!.,;:) ")
        thank_you_phrases = [
            "danke", "dankeschön", "dankeschoen", "vielen dank", "besten dank",
            "danke schön", "danke sehr",
            "thanks", "thank you", "thx", "ty",
            "dziękuję", "dziekuje", "dzięki", "dzienki", "dzieki",
            "gracias", "merci", "grazie", "bedankt", "dank u",
            "mulțumesc", "multumesc", "ačiū", "aciu",
            "děkuji", "dekuji", "ďakujem", "dakujem",
            "köszönöm", "koszonom",
            "hvala", "tack", "kiitos",
            "спасибо", "благодаря",
            "ok", "okay", "super", "alles klar", "perfekt", "perfect",
            "top", "great", "gut", "prima", "passt",
        ]
        # Only auto-close if the ENTIRE message is just a thank-you phrase
        # (exact match after stripping punctuation). This prevents closing
        # messages like "Danke, aber ich habe noch eine Frage..."
        is_pure_thanks = last_text_clean in thank_you_phrases
        if is_pure_thanks:
            logger.info(f"Ticket {ticket_id} follow-up is just a thank-you ({len(last_text)} chars) — closing without AI")
            from zendesk_api import _get_client, _base_url, _auth
            client = _get_client()
            await client.put(
                f"{_base_url()}/tickets/{ticket_id}.json",
                auth=_auth(),
                json={"ticket": {"status": "solved"}},
                timeout=10.0,
            )
            await add_tag(ticket_id, current_marker)
            return {"status": "ok", "ticket_id": ticket_id, "auto_closed": True, "reason": "thank_you_followup"}

        # --- Punkt 11: Exclude internal notes from LLM context ---
        conversation = []
        for c in comments:
            is_requester = c.get("author_id") == requester_id
            is_public = c.get("public", True)
            text = c.get("plain_body") or c.get("body", "")
            # Skip ALL internal notes — prevents leaking agent-to-agent comms
            if not is_public:
                continue
            if is_requester:
                conversation.append({"role": "Carrier", "text": text})
            else:
                conversation.append({"role": "Agent", "text": text})

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
            # --- Punkt 1: Validate before sending ---
            if reply_text and _validate_auto_reply(analysis, reply_text):
                logger.info(f"Ticket {ticket_id} follow-up auto-reply (AI flagged AUTO-REPLY: JA)")
                try:
                    await post_public_reply(ticket_id, reply_text, status="solved")
                    await add_tag(ticket_id, idempotency_marker)
                    await add_tag(ticket_id, "ai_auto_replied")
                    return {"status": "ok", "ticket_id": ticket_id, "auto_replied": True}
                except Exception as e:
                    logger.error(f"Failed to send follow-up auto-reply for ticket {ticket_id}: {e}")
            elif reply_text:
                logger.warning(f"Ticket {ticket_id} auto-reply blocked by safety validation")

        # No auto-reply — post as private note for agent
        note_body = f"🤖 AI Carrier Support Assistant (Follow-up)\n\n{analysis}"
        success = await post_private_note(ticket_id, note_body)
        await add_tag(ticket_id, idempotency_marker)
    else:
        # Run AI analysis
        try:
            analysis = await analyze_ticket(subject, message_body, requester_name)
        except Exception as e:
            logger.error(f"AI analysis failed for ticket {ticket_id}: {e}")
            return {"status": "error", "reason": "ai analysis failed"}

        # Extract category
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

        is_auto_reply = "AUTO-REPLY: JA" in analysis
        import unicodedata
        category = unicodedata.normalize("NFC", category)
        category = category.replace("ä", "ae").replace("ü", "ue").replace("ö", "oe")

        # --- Punkt 9: Bewerbung detection improved ---
        # AI category takes priority; keyword check as fallback only
        is_bewerbung_by_ai = "bewerbung" in category
        if not is_bewerbung_by_ai:
            # Keyword fallback with word-boundary-like matching
            bewerbung_keywords = [
                "bewerbung", "bewerben", "lebenslauf", "stellenangebot",
                "curriculum vitae", "job application", "job offer",
                "candidatura", "solicitud de empleo", "oferta de trabajo",
                "candidature", "offre d'emploi", "lettre de motivation",
                "offerta di lavoro", "domanda di lavoro",
                "sollicitatie", "vacature",
                "podanie o prace", "oferta pracy",
                "állásjelentkezés", "önéletrajz",
                "žádost o práci", "životopis",
                "cerere de angajare", "locuri de muncă",
                "jobbsøknad", "jobbansökan",
                "darbo paraiška", "tööavaldus", "darba pieteikums",
                "αίτηση εργασίας", "βιογραφικό",
                "кандидатура", "автобиография",
            ]
            text_lower = f"{subject} {message_body}".lower()
            is_bewerbung = any(kw in text_lower for kw in bewerbung_keywords)
        else:
            is_bewerbung = True

        if is_bewerbung:
            logger.info(f"Ticket {ticket_id} detected as Bewerbung — auto-reply")
            try:
                auto_reply = await generate_bewerbung_reply(subject, message_body)
                # --- Punkt 1: Validate ---
                if _validate_auto_reply("BEWERBUNG", auto_reply):
                    await post_public_reply(ticket_id, auto_reply, status="solved")
                    await add_tag(ticket_id, idempotency_marker)
                    await add_tag(ticket_id, "ai_auto_replied")
                    return {"status": "ok", "ticket_id": ticket_id, "auto_replied": True}
                else:
                    logger.warning(f"Ticket {ticket_id} Bewerbung auto-reply blocked by safety validation")
            except Exception as e:
                logger.error(f"Failed to send Bewerbung auto-reply for ticket {ticket_id}: {e}")

        elif is_auto_reply:
            reply_text = ""
            for i, line in enumerate(lines):
                if "ANTWORTVORSCHLAG" in line:
                    reply_text = "\n".join(lines[i + 1:]).strip()
                    break
            # --- Punkt 1: Validate ---
            if reply_text and _validate_auto_reply(analysis, reply_text):
                keep_open_categories = {"app/technik"}
                reply_status = "open" if category in keep_open_categories else "solved"
                logger.info(f"Ticket {ticket_id} category '{category}' — auto-reply (status={reply_status})")
                try:
                    await post_public_reply(ticket_id, reply_text, status=reply_status)
                    await add_tag(ticket_id, idempotency_marker)
                    await add_tag(ticket_id, "ai_auto_replied")
                    return {"status": "ok", "ticket_id": ticket_id, "auto_replied": True}
                except Exception as e:
                    logger.error(f"Failed to send auto-reply for ticket {ticket_id}: {e}")
            elif reply_text:
                logger.warning(f"Ticket {ticket_id} auto-reply blocked by safety validation")

        # No auto-reply — post as private note for agent
        note_body = f"🤖 AI Carrier Support Assistant\n\n{analysis}"
        success = await post_private_note(ticket_id, note_body)
        await add_tag(ticket_id, idempotency_marker)

    return {"status": "ok" if success else "error", "ticket_id": ticket_id, "auto_replied": False}
