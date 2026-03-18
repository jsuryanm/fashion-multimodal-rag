"""
utils/response_formatter.py
============================
Post-process the LLM's raw Markdown output for Gradio display.

Problems this solves:
  1. Gradio renders bare $ as LaTeX math delimiters — prices break.
  2. LLMs occasionally use inconsistent header styles (ALL CAPS, etc.).
  3. Vision models sometimes refuse fashion analysis with a safety message.
     We recover gracefully by extracting any item list that was included.
  4. Responses may not start with a heading — we add one for consistency.
"""

import logging
import re

logger = logging.getLogger(__name__)

# Phrases that indicate the model refused to analyse the image
_REFUSAL_PHRASES = [
    "I'm not able to provide",
    "I cannot provide",
    "I apologize, but I cannot",
    "I don't feel comfortable",
    "violated our content policy",
    "I'm unable to",
]

# Map raw LLM section names → clean Markdown headings
_HEADER_MAP = {
    "MATCHED CATALOGUE ITEMS:": "## Matched Catalogue Items",
    "SIMILAR CATALOGUE ITEMS:": "## Similar Catalogue Items",
    "ITEM DETAILS:": "## Item Details",
    "SIMILAR ITEMS:": "## Similar Items",
}


def format_response(response: str) -> str:
    """
    Clean and normalise the LLM's Markdown output.

    Steps:
      1. Handle empty response.
      2. Detect refusals — try to salvage any item list, else return fallback.
      3. Escape $ signs so Gradio doesn't render them as LaTeX.
      4. Normalise section headers.
      5. Ensure response starts with a top-level heading.
      6. Normalise bullet point styles (* → -).

    Args:
        response: Raw string from StrOutputParser.

    Returns:
        Clean Markdown string safe for Gradio display.
    """
    if not response or not response.strip():
        logger.warning("Empty response from LLM")
        return "# Fashion Analysis\n\nNo analysis was generated. Please try again."

    # ── Refusal handling ──────────────────────────────────────────────────────
    if any(phrase in response for phrase in _REFUSAL_PHRASES):
        logger.warning("LLM refusal detected — attempting item list recovery")
        for marker in _HEADER_MAP:
            if marker in response:
                section = response.split(marker, 1)[1].strip()
                return (
                    "# Fashion Analysis\n\n"
                    "The analysis could not be completed, but here are the matched items:\n\n"
                    f"## Catalogue Items\n\n{section}"
                )
        return (
            "# Fashion Analysis\n\n"
            "Analysis could not be completed for this image. "
            "Please try a different fashion photograph."
        )

    # ── Normal processing ─────────────────────────────────────────────────────

    # 1. Escape $ to prevent LaTeX rendering (prices like $89.99)
    processed = response.replace("$", "\\$")

    # 2. Normalise section headers
    for raw, clean in _HEADER_MAP.items():
        processed = processed.replace(raw, clean)

    # 3. Ensure top-level heading
    if not processed.lstrip().startswith("#"):
        processed = "# Fashion Analysis\n\n" + processed

    # 4. Normalise bullet point style (* → -)
    processed = re.sub(r"^\* ", "- ", processed, flags=re.MULTILINE)

    return processed