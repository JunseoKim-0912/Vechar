"""Convert assistant action syntax into semantic text for model history/memory."""

import re

_ACTION = re.compile(r"<action>(.*?)</action>", re.I | re.S)


def normalize_assistant_actions(content: str) -> str:
    def replace(match: re.Match[str]) -> str:
        action = match.group(1).strip()
        return f"[Character action: {action}]" if action else ""

    normalized = _ACTION.sub(replace, content)
    return re.sub(r"</?action>", "", normalized, flags=re.I)
