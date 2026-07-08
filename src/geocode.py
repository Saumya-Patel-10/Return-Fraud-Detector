"""
geocode.py
==========
Section 4 of the project plan: address normalization so "123 Main St" and
"123 Main Street" resolve to the same address_hash node.

Uses the Google Geocoding API when GOOGLE_API_KEY is set in .env.
Falls back to an offline regex-based normalizer when it isn't, so the
pipeline still runs end-to-end with zero external calls / no billing
account required.

Per the plan: we NEVER store the raw address or lat/lng together with a
customer identifier. We only ever persist address_hash, a SHA-256 hash
of the normalized address string. This mirrors how a real risk team
handles PII.
"""
from __future__ import annotations

import hashlib
import os
import re
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()

_API_KEY = os.environ.get("GOOGLE_API_KEY", "").strip()
_gmaps_client = None

_ABBREVIATIONS = {
    r"\bst\b": "street",
    r"\bave\b": "avenue",
    r"\bblvd\b": "boulevard",
    r"\brd\b": "road",
    r"\bdr\b": "drive",
    r"\bln\b": "lane",
    r"\bapt\b": "apartment",
    r"\bste\b": "suite",
}


def _get_client():
    global _gmaps_client
    if _gmaps_client is None and _API_KEY:
        import googlemaps  # local import so the package is optional
        _gmaps_client = googlemaps.Client(key=_API_KEY)
    return _gmaps_client


def _offline_normalize(address: str) -> str:
    """
    Regex-based normalizer used when no Google API key is configured.
    Lowercases, strips punctuation, and expands common street
    abbreviations so trivial string variants collapse to the same value.
    """
    s = address.lower().strip()
    s = re.sub(r"[.,]", "", s)
    s = re.sub(r"\s+", " ", s)
    for pattern, replacement in _ABBREVIATIONS.items():
        s = re.sub(pattern, replacement, s)
    return s


@lru_cache(maxsize=100_000)
def normalize_address(address: str) -> str:
    """
    Return a normalized address string. Tries the Google Geocoding API
    first (if configured), falls back to the offline normalizer on any
    failure (missing key, quota, network error) so the pipeline never
    hard-fails on geocoding.
    """
    client = _get_client()
    if client is not None:
        try:
            result = client.geocode(address)
            if result:
                return result[0]["formatted_address"].lower().strip()
        except Exception:
            pass  # fall through to offline normalization
    return _offline_normalize(address)


def address_hash(address: str) -> str:
    """
    SHA-256 hash of the normalized address. This is the only address
    representation that should ever be persisted or joined against
    other tables -- never store raw address or lat/lng next to a
    customer_id.
    """
    normalized = normalize_address(address)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


if __name__ == "__main__":
    samples = ["123 Main St, Dallas, TX 75201", "123 Main Street, Dallas, TX 75201"]
    for s in samples:
        print(s, "->", address_hash(s), f"(using {'Google API' if _API_KEY else 'offline normalizer'})")
