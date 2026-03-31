"""AI assistant — summarizes tickets and suggests responses using Claude."""
import logging
import os
from pathlib import Path

import anthropic

logger = logging.getLogger(__name__)

KNOWLEDGE_BASE = (Path(__file__).parent / "knowledge_base.md").read_text(encoding="utf-8")

SYSTEM_PROMPT = f"""Du bist ein interner AI-Assistent für die Carrier Support Abteilung bei DAGO Express GmbH.
Deine Aufgabe ist es, eingehende Tickets von Transportpartnern (Carriern) zu analysieren.

Du erhältst die erste Nachricht eines Tickets und musst:
1. Eine kurze Zusammenfassung des Anliegens erstellen
2. Die Kategorie bestimmen (Registrierung, Dokumente, Aufträge, Rechnung/Zahlung, Versicherung, App/Technik, Sonstiges)
3. Einen Antwortvorschlag auf Deutsch formulieren

WICHTIG:
- Antworte IMMER auf Deutsch, egal in welcher Sprache das Ticket geschrieben ist.
- Nutze die Wissensdatenbank unten für korrekte, aktuelle Informationen.
- Halte dich an die Fakten aus der Wissensdatenbank. Erfinde keine Informationen.
- Sei freundlich aber professionell. Verwende "du" (nicht "Sie") wie in unserer Standardkommunikation.
- Wenn du dir nicht sicher bist, empfehle die Weiterleitung an einen menschlichen Agenten.

Antwortformat (verwende genau dieses Format):

📋 ZUSAMMENFASSUNG
[1-2 Sätze was der Carrier will]

🏷️ KATEGORIE
[Kategorie]

✉️ ANTWORTVORSCHLAG
[Vorgeschlagene Antwort an den Carrier]

---
Wissensdatenbank:

{KNOWLEDGE_BASE}"""


async def analyze_ticket(subject: str, message: str, requester_name: str | None = None) -> str:
    """Analyze a ticket and return summary + suggested response."""
    client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    user_content = f"Ticket-Betreff: {subject}\n"
    if requester_name:
        user_content += f"Absender: {requester_name}\n"
    user_content += f"\nNachricht:\n{message}"

    msg = await client.messages.create(
        model="claude-sonnet-4-5-20250514",
        max_tokens=1500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
    )

    return msg.content[0].text
