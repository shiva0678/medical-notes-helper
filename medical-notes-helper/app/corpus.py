"""Loading and preprocessing for the synthetic administrative text corpus."""

import re
from collections.abc import Iterable
from pathlib import Path

_NON_WORD_CHARACTERS = re.compile(r"[^a-z0-9\s]")
_DEFAULT_CORPUS_PATH = (
    Path(__file__).resolve().parent.parent
    / "corpus"
    / "administrative_medical_sentences.txt"
)


def preprocess_sentence(sentence: str) -> list[str]:
    """Lowercase a sentence, replace punctuation with spaces, and tokenize it."""
    normalized = _NON_WORD_CHARACTERS.sub(" ", sentence.lower())
    return normalized.split()


def preprocess_corpus(sentences: Iterable[str]) -> list[list[str]]:
    """Normalize sentences and remove empty entries and cleaned duplicates."""
    processed: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()

    for sentence in sentences:
        tokens = preprocess_sentence(sentence)
        key = tuple(tokens)
        if tokens and key not in seen:
            processed.append(tokens)
            seen.add(key)

    return processed


def load_corpus(path: str | Path | None = None) -> list[list[str]]:
    """Load and preprocess the bundled synthetic corpus or a supplied text file."""
    corpus_path = Path(path) if path is not None else _DEFAULT_CORPUS_PATH
    return preprocess_corpus(corpus_path.read_text(encoding="utf-8").splitlines())
