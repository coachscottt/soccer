import os
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
load_dotenv()

TELEGRAM_API = "https://api.telegram.org"


def send_telegram_message(text: str,
                          bot_token: str | None = None,
                          chat_id: str | None = None,
                          parse_mode: str = "Markdown") -> bool:
    """
    Send a message via the Telegram Bot API.
    Returns True on success; no-ops with a warning when the bot
    is not configured (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID).

    Telegram messages cap at 4096 chars; longer texts are split
    on line boundaries.
    """
    bot_token = bot_token or os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID")
    if not bot_token or not chat_id:
        print("  [WARN] Telegram not configured "
              "(TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID missing) - skipping send")
        return False

    chunks = _split_message(text)
    for chunk in chunks:
        try:
            resp = requests.post(
                f"{TELEGRAM_API}/bot{bot_token}/sendMessage",
                json={"chat_id": chat_id, "text": chunk,
                      "parse_mode": parse_mode},
                timeout=15,
            )
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"  [WARN] Telegram send failed: {e}")
            return False
    return True


def _split_message(text: str, limit: int = 4096) -> list[str]:
    """Split text into <=limit chunks, preferring line boundaries."""
    if len(text) <= limit:
        return [text]
    chunks, current = [], ""
    for line in text.splitlines(keepends=True):
        if len(current) + len(line) > limit:
            if current:
                chunks.append(current)
            # a single line longer than limit gets hard-split
            while len(line) > limit:
                chunks.append(line[:limit])
                line = line[limit:]
            current = line
        else:
            current += line
    if current:
        chunks.append(current)
    return chunks
