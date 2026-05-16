"""Custom action: a tiny keyword-based toxicity check.

In a real deployment you would call a moderation API or a classifier model
(e.g. NVIDIA NeMo Curator, Detoxify, OpenAI moderation). This minimal stub
keeps the lab runnable offline and makes the wiring clear.

Colang 2 action-naming convention:
  - action name is CamelCase and ends with "Action"
  - called from .co files with `await ToxicityCheckAction(text=...)`
"""

from typing import Optional

from nemoguardrails.actions import action


_BAD_WORDS = {
    "idiot",
    "stupid",
    "hate you",
    "kill",
    "moron",
}


@action(name="ToxicityCheckAction")
async def toxicity_check(text: Optional[str] = None) -> bool:
    """Return True when the text looks toxic."""
    if not text:
        return False
    lowered = text.lower()
    return any(word in lowered for word in _BAD_WORDS)
