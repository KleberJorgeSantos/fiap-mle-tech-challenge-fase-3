"""Text normalization shared by training and serving (stdlib only, no pandas)."""

import re
import unicodedata

_WHITESPACE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    r"""Normalize a report: ASCII-folded, lowercase, trimmed, single spaces.

    The API applies the exact same transformation before calling either backend.
    ASCII folding (accented letters -> plain letters) guarantees the ONNX tokenizer
    (ASCII regex) and sklearn (unicode ``\w``) produce identical tokens.
    """
    folded = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode("ascii")
    return _WHITESPACE.sub(" ", folded).strip().lower()
