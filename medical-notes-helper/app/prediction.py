"""Context-aware next-word prediction from observed n-grams."""

from collections.abc import Sequence
from dataclasses import dataclass

from app.corpus import preprocess_sentence
from app.language_model import NGramLanguageModel


@dataclass(frozen=True)
class Suggestion:
    """A candidate next word and its conditional probability."""

    word: str
    probability: float


def _tokenize_input(text: str | Sequence[str]) -> list[str]:
    if isinstance(text, str):
        return preprocess_sentence(text)
    if isinstance(text, (bytes, bytearray)) or not isinstance(text, Sequence):
        raise TypeError("input must be a sentence or a sequence of words")
    if any(not isinstance(token, str) for token in text):
        raise TypeError("every input word must be a string")
    return preprocess_sentence(" ".join(text))


def _trigram_candidates(
    model: NGramLanguageModel, context: tuple[str, str]
) -> set[str]:
    return {
        following
        for (first, second, following), count in model.trigram_counts.items()
        if (first, second) == context and count > 0
    }


def _bigram_candidates(model: NGramLanguageModel, context: str) -> set[str]:
    return {
        following
        for (previous, following), count in model.bigram_counts.items()
        if previous == context and count > 0
    }


def _select_candidates(
    model: NGramLanguageModel, tokens: list[str]
) -> tuple[set[str], tuple[str, ...]]:
    if len(tokens) >= 2:
        trigram_context = (tokens[-2], tokens[-1])
        candidates = _trigram_candidates(model, trigram_context)
        if candidates:
            return candidates, trigram_context

    if tokens:
        bigram_context = tokens[-1]
        candidates = _bigram_candidates(model, bigram_context)
        if candidates:
            return candidates, (bigram_context,)

    candidates = {
        word for word, count in model.unigram_counts.items() if count > 0
    }
    return candidates, ()


def suggest_next_words(
    model: NGramLanguageModel,
    text: str | Sequence[str],
    top_k: int = 3,
) -> list[Suggestion]:
    """Suggest up to ``top_k`` observed next words, ranked by probability.

    The predictor uses the longest context with observed continuations and
    backs off from trigram to bigram to unigram when a context is unavailable.
    """
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise TypeError("top_k must be an integer")
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    tokens = _tokenize_input(text)
    candidates, context = _select_candidates(model, tokens)
    suggestions = [
        Suggestion(
            word=word,
            probability=model.conditional_probability(word, context),
        )
        for word in candidates
    ]
    suggestions.sort(key=lambda suggestion: (-suggestion.probability, suggestion.word))
    return suggestions[:top_k]
