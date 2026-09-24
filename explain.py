"""Stage-2 prompt and robust parsing of the VLM's JSON explanation (pure Python)."""
from __future__ import annotations

import json
import math

PROMPT_JSON = (
    "You are a retail security assistant. Describe the event in the image as JSON with "
    "keys: anomaly_type, description, confidence, recommended_action, supporting_visual_cue."
)
KEYS = ("anomaly_type", "description", "confidence", "recommended_action",
        "supporting_visual_cue")


def _first_json_object(text: str):
    """Return the first balanced {...} substring of `text` that parses as JSON, or None."""
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


def parse_explanation(text: str) -> dict:
    """Turn the raw generated text into a dict with all of KEYS present.

    The VLM is only asked to emit JSON; small models often add prose or malformed JSON.
    On failure the raw text becomes `description` and `confidence` is NaN, so downstream
    code never crashes on a string (the original evaluation loop called `.get` on a str).
    """
    obj = _first_json_object(text) if text else None
    out = {k: None for k in KEYS}
    if isinstance(obj, dict):
        out.update({k: obj.get(k) for k in KEYS})
        out["parsed"] = True
    else:
        out["description"] = (text or "").strip()
        out["parsed"] = False
    try:
        c = float(out["confidence"])
        out["confidence"] = c if math.isfinite(c) else float("nan")
    except (TypeError, ValueError):
        out["confidence"] = float("nan")
    return out
