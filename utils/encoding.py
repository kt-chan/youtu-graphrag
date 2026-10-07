"""Byte → string decoding with encoding detection and robust fallbacks."""
from typing import Optional


def _detect_encoding_from_bytes(data: bytes) -> Optional[str]:
    """Detect encoding via chardet if available; return lower-cased name or None."""
    try:
        import chardet  # type: ignore

        result = chardet.detect(data) or {}
        enc = result.get("encoding")
        if enc:
            return enc.lower()
    except Exception:
        pass
    return None


def decode_bytes_with_detection(data: bytes) -> str:
    """Decode bytes with encoding detection and robust fallbacks.

    Order: detected → utf-8/utf-8-sig → common Chinese encodings →
    utf-16 variants → latin-1 → replace.
    """
    candidates = []
    detected = _detect_encoding_from_bytes(data)
    if detected:
        candidates.append(detected)
    candidates.extend(
        [
            "utf-8",
            "utf-8-sig",
            "gb18030",
            "gbk",
            "big5",
            "utf-16",
            "utf-16le",
            "utf-16be",
            "latin-1",
        ]
    )
    tried = set()
    for enc in candidates:
        if not enc or enc in tried:
            continue
        tried.add(enc)
        try:
            return data.decode(enc)
        except Exception:
            continue
    return data.decode("utf-8", errors="replace")