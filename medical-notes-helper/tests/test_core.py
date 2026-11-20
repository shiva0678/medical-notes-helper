"""Focused tests for the milestone implementations."""

import pytest

# Milestone 1: Curate an anonymized corpus of administrative medical text.
from app.core import milestone_1, milestone_2
from app.corpus import preprocess_corpus, preprocess_sentence
from app.language_model import NGramLanguageModel


def test_preprocess_sentence_normalizes_case_punctuation_and_whitespace():
    assert preprocess_sentence("  Follow-UP, VISIT  scheduled! ") == [
        "follow",
        "up",
        "visit",
        "scheduled",
    ]


def test_preprocess_corpus_removes_empty_entries_and_cleaned_duplicates():
    sentences = [
        "Referral confirmed!",
        "   ",
        " referral   confirmed ",
        "!!!",
        "Referral sent to clinic.",
    ]

    assert preprocess_corpus(sentences) == [
        ["referral", "confirmed"],
        ["referral", "sent", "to", "clinic"],
    ]


def test_milestone_1_loads_diverse_clean_synthetic_corpus():
    corpus = milestone_1()
    words = {word for sentence in corpus for word in sentence}
    required_categories = {
        "appointment",
        "registration",
        "record",
        "referral",
        "discharge",
        "insurance",
        "billing",
    }

    assert len(corpus) >= 60
    assert all(
        sentence
        and all(word.isalnum() and word == word.lower() for word in sentence)
        for sentence in corpus
    )
    assert required_categories.issubset(words)


def test_milestone_2_trains_on_curated_corpus():
    model = milestone_2()

    assert model.vocabulary_size > 0
    assert model.total_tokens == sum(len(sentence) for sentence in milestone_1())


def test_ngram_model_counts_unigrams_bigrams_and_trigrams():
    model = NGramLanguageModel().fit(
        [["the", "patient", "waited"], ["the", "patient", "called"]]
    )

    assert model.unigram_counts == {
        "the": 2,
        "patient": 2,
        "waited": 1,
        "called": 1,
    }
    assert model.bigram_counts[("the", "patient")] == 2
    assert model.bigram_counts[("patient", "waited")] == 1
    assert model.trigram_counts[("the", "patient", "waited")] == 1
    assert model.vocabulary_size == 4
    assert model.total_tokens == 6


def test_conditional_probabilities_use_unigram_bigram_and_trigram_contexts():
    model = NGramLanguageModel().fit(
        [["the", "patient", "waited"], ["the", "patient", "called"]]
    )

    assert model.conditional_probability("the") == pytest.approx(3 / 10)
    assert model.conditional_probability("patient", ("the",)) == pytest.approx(3 / 6)
    assert model.conditional_probability(
        "waited", ("the", "patient")
    ) == pytest.approx(2 / 6)


def test_laplace_smoothing_handles_unseen_ngrams_and_unknown_words():
    model = NGramLanguageModel().fit(
        [["the", "patient", "waited"], ["the", "patient", "called"]]
    )
    trained_counts = (
        model.unigram_counts.copy(),
        model.bigram_counts.copy(),
        model.trigram_counts.copy(),
    )

    assert model.conditional_probability("unseen", ("the",)) == pytest.approx(1 / 6)
    assert model.conditional_probability("called", ("unknown",)) == pytest.approx(1 / 4)
    assert model.conditional_probability(
        "unseen", ("unknown", "context")
    ) == pytest.approx(1 / 4)
    assert model.conditional_probability("the", ("patient",)) == pytest.approx(1 / 6)

    known_word_probabilities = [
        model.conditional_probability(word, ("unknown",))
        for word in model.vocabulary
    ]
    assert sum(known_word_probabilities) == pytest.approx(1.0)
    assert all(0 <= probability <= 1 for probability in known_word_probabilities)
    assert trained_counts == (
        model.unigram_counts,
        model.bigram_counts,
        model.trigram_counts,
    )


def test_empty_corpus_returns_empty_counts_and_zero_probabilities():
    model = NGramLanguageModel().fit([])

    assert model.unigram_counts == {}
    assert model.bigram_counts == {}
    assert model.trigram_counts == {}
    assert model.vocabulary_size == 0
    assert model.conditional_probability("word") == 0.0
    assert model.conditional_probability("word", ("context",)) == 0.0


def test_invalid_training_and_probability_inputs_raise_clear_errors():
    model = NGramLanguageModel()

    with pytest.raises(TypeError, match="tokenized sentences"):
        model.fit("not tokenized")
    with pytest.raises(TypeError, match="iterable of tokens"):
        model.fit(["not tokenized"])
    with pytest.raises(TypeError, match="string"):
        model.fit([["valid", 1]])
    with pytest.raises(ValueError, match="must not be empty"):
        model.fit([[""]])
    with pytest.raises(ValueError, match="at most two"):
        model.conditional_probability("word", ("one", "two", "three"))
    with pytest.raises(TypeError, match="iterable of tokens"):
        model.conditional_probability("word", "not a token sequence")
    with pytest.raises(ValueError, match="word must not be empty"):
        model.conditional_probability(" ")


# Milestone 3: Develop a text generation feature that suggests words based on input.
@pytest.mark.skip(reason="not started")
def test_milestone_3():
    raise NotImplementedError("write the test for milestone 3")


# Milestone 4: Create a CLI for users to interact with the suggestion engine.
@pytest.mark.skip(reason="not started")
def test_milestone_4():
    raise NotImplementedError("write the test for milestone 4")

