"""Focused tests for the milestone implementations."""

import pytest

# Milestone 1: Curate an anonymized corpus of administrative medical text.
from app.core import milestone_1
from app.corpus import preprocess_corpus, preprocess_sentence


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


# Milestone 2: Implement the n-gram language model and apply smoothing.
@pytest.mark.skip(reason="not started")
def test_milestone_2():
    raise NotImplementedError("write the test for milestone 2")


# Milestone 3: Develop a text generation feature that suggests words based on input.
@pytest.mark.skip(reason="not started")
def test_milestone_3():
    raise NotImplementedError("write the test for milestone 3")


# Milestone 4: Create a CLI for users to interact with the suggestion engine.
@pytest.mark.skip(reason="not started")
def test_milestone_4():
    raise NotImplementedError("write the test for milestone 4")

