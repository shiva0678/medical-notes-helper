"""A small unigram, bigram, and trigram language model."""

from collections import Counter
from collections.abc import Iterable


class NGramLanguageModel:
    """Count n-grams and calculate Laplace-smoothed conditional probabilities."""

    def __init__(self) -> None:
        self.unigram_counts: Counter[str] = Counter()
        self.bigram_counts: Counter[tuple[str, str]] = Counter()
        self.trigram_counts: Counter[tuple[str, str, str]] = Counter()
        self._bigram_context_counts: Counter[str] = Counter()
        self._trigram_context_counts: Counter[tuple[str, str]] = Counter()
        self.vocabulary: set[str] = set()
        self.total_tokens = 0

    @property
    def vocabulary_size(self) -> int:
        """Return the number of unique tokens observed during training."""
        return len(self.vocabulary)

    def fit(self, sentences: Iterable[Iterable[str]]) -> "NGramLanguageModel":
        """Train on tokenized sentences, replacing any previously learned counts."""
        if isinstance(sentences, (str, bytes)):
            raise TypeError("sentences must be an iterable of tokenized sentences")

        unigram_counts: Counter[str] = Counter()
        bigram_counts: Counter[tuple[str, str]] = Counter()
        trigram_counts: Counter[tuple[str, str, str]] = Counter()
        bigram_context_counts: Counter[str] = Counter()
        trigram_context_counts: Counter[tuple[str, str]] = Counter()
        vocabulary: set[str] = set()
        total_tokens = 0

        try:
            sentence_iterator = iter(sentences)
        except TypeError as error:
            raise TypeError(
                "sentences must be an iterable of tokenized sentences"
            ) from error

        for sentence in sentence_iterator:
            if isinstance(sentence, (str, bytes)):
                raise TypeError("each sentence must be an iterable of tokens")

            try:
                tokens = tuple(sentence)
            except TypeError as error:
                raise TypeError(
                    "each sentence must be an iterable of tokens"
                ) from error

            if any(not isinstance(token, str) for token in tokens):
                raise TypeError("every token must be a string")
            normalized_tokens = tuple(token.lower() for token in tokens)
            if any(not token.strip() for token in normalized_tokens):
                raise ValueError("tokens must not be empty or whitespace")

            unigram_counts.update(normalized_tokens)
            bigram_counts.update(zip(normalized_tokens, normalized_tokens[1:]))
            trigram_counts.update(
                zip(
                    normalized_tokens,
                    normalized_tokens[1:],
                    normalized_tokens[2:],
                )
            )
            bigram_context_counts.update(normalized_tokens[:-1])
            trigram_context_counts.update(
                zip(normalized_tokens[:-2], normalized_tokens[1:-1])
            )
            vocabulary.update(normalized_tokens)
            total_tokens += len(normalized_tokens)

        self.unigram_counts = unigram_counts
        self.bigram_counts = bigram_counts
        self.trigram_counts = trigram_counts
        self._bigram_context_counts = bigram_context_counts
        self._trigram_context_counts = trigram_context_counts
        self.vocabulary = vocabulary
        self.total_tokens = total_tokens
        return self

    def conditional_probability(
        self, word: str, context: Iterable[str] = ()
    ) -> float:
        """Return P(word | context) with add-one smoothing for up to two tokens."""
        if not isinstance(word, str):
            raise TypeError("word must be a string")
        normalized_word = word.strip().lower()
        if not normalized_word:
            raise ValueError("word must not be empty or whitespace")
        if isinstance(context, (str, bytes)):
            raise TypeError("context must be an iterable of tokens")

        try:
            context_tokens = tuple(context)
        except TypeError as error:
            raise TypeError("context must be an iterable of tokens") from error

        if len(context_tokens) > 2:
            raise ValueError("context must contain at most two tokens")
        if any(not isinstance(token, str) for token in context_tokens):
            raise TypeError("every context token must be a string")

        normalized_context = tuple(token.strip().lower() for token in context_tokens)
        if any(not token for token in normalized_context):
            raise ValueError("context tokens must not be empty or whitespace")

        vocabulary_size = self.vocabulary_size
        if vocabulary_size == 0:
            return 0.0

        if not normalized_context:
            count = self.unigram_counts.get(normalized_word, 0)
            context_count = self.total_tokens
        elif len(normalized_context) == 1:
            context_token = normalized_context[0]
            count = self.bigram_counts.get((context_token, normalized_word), 0)
            context_count = self._bigram_context_counts.get(context_token, 0)
        else:
            context_pair = (normalized_context[0], normalized_context[1])
            count = self.trigram_counts.get(
                (context_pair[0], context_pair[1], normalized_word), 0
            )
            context_count = self._trigram_context_counts.get(context_pair, 0)

        return (count + 1) / (context_count + vocabulary_size)
