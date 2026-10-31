"""Explicit online drug-reference lookup with a local cache."""

import json
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

CACHE_PATH = Path("models/online_reference.json")


@dataclass(frozen=True)
class OnlineLookupResult:
    """Verified ingredient result returned by an approved provider."""

    query: str
    display_name: str
    ingredients: list[str]
    provider: str
    citation_url: str


def _open_json(url: str) -> dict:
    """Fetch one JSON API response after explicit user approval."""
    with urlopen(url, timeout=15) as response:
        return json.load(response)


def lookup_openfda(query: str) -> OnlineLookupResult | None:
    """Look up active ingredients in the openFDA label API."""
    url = (
        "https://api.fda.gov/drug/label.json?"
        f"search=openfda.brand_name:%22{quote(query)}%22&limit=1"
    )
    try:
        payload = _open_json(url)
    except (HTTPError, URLError, TimeoutError, ValueError):
        return None
    results = payload.get("results", [])
    if not results:
        return None
    record = results[0]
    ingredients = record.get("active_ingredient", [])
    if not ingredients:
        ingredients = record.get("openfda", {}).get("substance_name", [])
    if not ingredients:
        ingredients = record.get("openfda", {}).get("generic_name", [])
    return OnlineLookupResult(
        query=query,
        display_name=record.get("openfda", {}).get("brand_name", [query])[0],
        ingredients=sorted(set(ingredients)),
        provider="openFDA Drug Label API",
        citation_url=url,
    )


def lookup_rxnorm(query: str) -> OnlineLookupResult | None:
    """Look up ingredient concepts through RxNorm."""
    search_url = (
        "https://rxnav.nlm.nih.gov/REST/approximateTerm.json?"
        f"term={quote(query)}&maxEntries=1"
    )
    try:
        payload = _open_json(search_url)
        candidates = payload.get("approximateGroup", {}).get("candidate", [])
        if not candidates:
            return None
        rxcui = candidates[0]["rxcui"]
        details_url = f"https://rxnav.nlm.nih.gov/REST/rxcui/{rxcui}/allrelated.json"
        details = _open_json(details_url)
    except (KeyError, IndexError, TypeError, ValueError, OSError):
        return None
    ingredients = {
        prop["name"]
        for group in details.get("allRelatedGroup", {}).get("conceptGroup", [])
        if group.get("tty") in {"IN", "PIN", "SCDC"}
        for prop in group.get("conceptProperties", [])
        if prop.get("name")
    }
    if not ingredients:
        return None
    return OnlineLookupResult(
        query=query,
        display_name=query,
        ingredients=sorted(ingredients),
        provider="RxNorm API",
        citation_url=details_url,
    )


def save_result(result: OnlineLookupResult) -> None:
    """Save an approved lookup for future offline matching."""
    entries = {}
    if CACHE_PATH.exists():
        entries = json.loads(CACHE_PATH.read_text())
    entries[result.display_name.lower()] = {
        "display_name": result.display_name,
        "ingredients": result.ingredients,
        "provider": result.provider,
        "citation_url": result.citation_url,
    }
    CACHE_PATH.write_text(json.dumps(entries, indent=2) + "\n")
