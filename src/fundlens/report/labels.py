"""Display labels for report-facing financial terms."""
from __future__ import annotations

import re
from typing import Any

FACTOR_LABELS = {
    "MKT_RF": "Market",
    "SMB": "Size",
    "HML": "Value",
    "RMW": "Profitability",
    "CMA": "Investment",
    "MOM": "Momentum",
    "RF": "Risk-free rate",
    "alpha": "Alpha",
}

_FACTOR_CODES_PATTERN = "|".join(
    re.escape(code) for code in FACTOR_LABELS if code.isupper()
)
_FACTOR_CODE_RE = re.compile(
    rf"(?<![+A-Za-z0-9_])({_FACTOR_CODES_PATTERN})(?![A-Za-z0-9_])"
)


def factor_label(code: object) -> str:
    """Return the human-readable label for a factor code."""
    text = str(code)
    return FACTOR_LABELS.get(text, text)


def factor_text(text: str) -> str:
    """Replace standalone factor codes in display text with readable labels."""
    return _FACTOR_CODE_RE.sub(lambda match: FACTOR_LABELS[match.group(1)], text)


def factor_display_value(value: Any) -> Any:
    """Recursively relabel factor-code strings in presentation data."""
    if isinstance(value, str):
        return factor_text(value)
    if isinstance(value, dict):
        return {key: factor_display_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [factor_display_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(factor_display_value(item) for item in value)
    return value
