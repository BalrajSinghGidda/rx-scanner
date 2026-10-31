"""Build an offline RxNorm drug-name vocabulary."""

import json
from pathlib import Path
from urllib.request import urlopen

VOCAB_PATH = Path("models/rxnorm_vocab.json")
RXNORM_URL = "https://rxnav.nlm.nih.gov/REST/allconcepts.json?tty=IN"


def fetch_rxnorm_names() -> list[str]:
    """Fetch ingredient names from the public RxNorm API."""
    with urlopen(RXNORM_URL, timeout=60) as response:
        payload = json.load(response)
    concepts = payload["minConceptGroup"]["minConcept"]
    return sorted(
        {
            concept["name"].strip()
            for concept in concepts
            if concept.get("name") and any(
                character.isalpha() for character in concept["name"]
            )
        }
    )


def main() -> None:
    """Fetch and save the vocabulary used offline by the app."""
    vocabulary = fetch_rxnorm_names()
    VOCAB_PATH.parent.mkdir(parents=True, exist_ok=True)
    VOCAB_PATH.write_text(json.dumps(vocabulary, indent=2) + "\n")
    print(f"Saved {len(vocabulary)} RxNorm names to {VOCAB_PATH}")


if __name__ == "__main__":
    main()
