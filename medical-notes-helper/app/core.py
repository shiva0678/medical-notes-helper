"""Core functionality, implemented one project milestone at a time."""

from app.corpus import load_corpus
from app.language_model import NGramLanguageModel

# --- Milestone 1: Curate an anonymized corpus of administrative medical text.
# Hint: Use text processing techniques to clean and prepare your corpus.
def milestone_1() -> list[list[str]]:
    """Return the bundled, cleaned synthetic administrative text corpus."""
    return load_corpus()


# --- Milestone 2: Implement the n-gram language model and apply smoothing.
# Hint: Implement the n-gram counting and probability calculation, then apply Laplace or Kneser-Ney smoothing.
def milestone_2() -> NGramLanguageModel:
    """Train and return the n-gram model using the bundled corpus."""
    return NGramLanguageModel().fit(milestone_1())


# --- Milestone 3: Develop a text generation feature that suggests words based on input.
# Hint: Develop a function that takes a sequence of words and returns the most probable next word.
def milestone_3():  # TODO(milestone 3)
    raise NotImplementedError("TODO(milestone 3)")


# --- Milestone 4: Create a CLI for users to interact with the suggestion engine.
# Hint: Use Python's `argparse` or `click` libraries to build an interactive command-line interface.
def milestone_4():  # TODO(milestone 4)
    raise NotImplementedError("TODO(milestone 4)")
