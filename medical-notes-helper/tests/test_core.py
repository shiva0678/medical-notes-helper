"""Focused tests for the milestone implementations."""

import pytest

# Milestone 1: Curate an anonymized corpus of administrative medical text.
from app.core import milestone_1, milestone_2, milestone_3
from app.corpus import preprocess_corpus, preprocess_sentence
from app.language_model import NGramLanguageModel
from app.prediction import Suggestion, suggest_next_words


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


def _prediction_model():
    return NGramLanguageModel().fit(
        [
            ["the", "patient", "was", "scheduled"],
            ["the", "patient", "was", "advised"],
            ["a", "patient", "arrived"],
            ["a", "patient", "arrived"],
            ["a", "patient", "arrived"],
            ["a", "patient", "arrived"],
            ["a", "patient", "arrived"],
            ["office", "records", "were", "reviewed"],
        ]
    )


def test_milestone_3_predicts_from_the_trained_corpus():
    suggestions = milestone_3("the patient")

    assert suggestions
    assert all(isinstance(suggestion, Suggestion) for suggestion in suggestions)
    assert all(suggestion.probability > 0 for suggestion in suggestions)


def test_prediction_prefers_trigram_context_over_more_common_bigram_continuation():
    suggestions = suggest_next_words(_prediction_model(), "the patient", top_k=5)

    assert len(suggestions) == 1
    assert suggestions[0].word == "was"
    assert suggestions[0].probability == pytest.approx(3 / 13)


def test_prediction_uses_bigram_when_trigram_context_is_unseen():
    suggestions = suggest_next_words(_prediction_model(), "unknown patient")

    assert suggestions[0].word == "arrived"
    assert suggestions[0].probability == pytest.approx(1 / 3)


def test_prediction_falls_back_to_unigrams_for_unknown_contexts():
    model = _prediction_model()

    unknown_word_suggestions = suggest_next_words(model, "unseen")
    unseen_pair_suggestions = suggest_next_words(model, ("not-in-corpus", "unseen"))

    assert unknown_word_suggestions == unseen_pair_suggestions
    assert unknown_word_suggestions[0].word == "patient"
    assert unknown_word_suggestions[0].probability == pytest.approx(8 / 38)


def test_prediction_returns_ranked_top_three_and_top_five():
    model = _prediction_model()
    top_three = suggest_next_words(model, "", top_k=3)
    top_five = suggest_next_words(model, [], top_k=5)

    assert len(top_three) == 3
    assert len(top_five) == 5
    assert top_three == top_five[:3]
    assert [item.probability for item in top_five] == sorted(
        (item.probability for item in top_five), reverse=True
    )


def test_prediction_breaks_probability_ties_alphabetically():
    model = NGramLanguageModel().fit([["zebra"], ["apple"], ["mango"]])

    suggestions = suggest_next_words(model, "")

    assert [suggestion.word for suggestion in suggestions] == [
        "apple",
        "mango",
        "zebra",
    ]


def test_prediction_does_not_fill_results_with_unobserved_words():
    suggestions = suggest_next_words(_prediction_model(), "the patient", top_k=5)

    assert [suggestion.word for suggestion in suggestions] == ["was"]


def test_prediction_normalizes_case_and_punctuation_and_handles_repeated_words():
    model = NGramLanguageModel().fit([["go", "go", "go"], ["go", "home"]])

    repeated_word_suggestions = suggest_next_words(model, ["GO", "go!"], top_k=5)
    punctuation_suggestions = suggest_next_words(
        _prediction_model(), "The patient,", top_k=5
    )

    assert [suggestion.word for suggestion in repeated_word_suggestions] == ["go"]
    assert [suggestion.word for suggestion in punctuation_suggestions] == ["was"]


def test_prediction_returns_no_results_without_trained_tokens():
    model = NGramLanguageModel().fit([])

    assert suggest_next_words(model, "any context", top_k=5) == []


def test_prediction_rejects_invalid_input_and_result_limit():
    model = _prediction_model()

    with pytest.raises(TypeError, match="sequence of words"):
        suggest_next_words(model, 42)
    with pytest.raises(TypeError, match="input word must be a string"):
        suggest_next_words(model, ["the", 42])
    with pytest.raises(ValueError, match="greater than zero"):
        suggest_next_words(model, "the patient", top_k=0)


# Milestone 4: Create a CLI for users to interact with the suggestion engine.
@pytest.mark.skip(reason="not started")
def test_milestone_4():
    raise NotImplementedError("write the test for milestone 4")

