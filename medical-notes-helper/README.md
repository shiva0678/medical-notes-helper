# Medical Notes Helper

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/ckmxkb47xb/badge.svg)](https://ask.thili.ai/b/ckmxkb47xb)

This CLI tool assists clinic staff in drafting administrative medical notes by offering next-word suggestions. By leveraging an n-gram language model trained on anonymized medical text, it aims to enhance consistency and efficiency in documentation, reducing the need for extensive manual typing.

## What you are building

**Goal.** You will build a command-line interface (CLI) tool that helps clinic staff draft administrative sections of medical notes by generating next-word suggestions, ensuring consistency and reducing manual typing. This tool will demonstrate the ability to generate text from a language model, providing context-aware suggestions.

**Done when.** The project is complete when you have curated an anonymized corpus of administrative medical text, implemented and smoothed an n-gram language model, developed a feature to suggest words based on user input, and created a functional CLI for users to interact with the suggestion engine.

## Milestones

Tick these off as you go, and commit after each one.

- [x] **1. Curate an anonymized corpus of administrative medical text.**
      Hint: Use text processing techniques to clean and prepare your corpus.
- [x] **2. Implement the n-gram language model and apply smoothing.**
      Hint: Implement the n-gram counting and probability calculation, then apply Laplace or Kneser-Ney smoothing.
- [x] **3. Develop a text generation feature that suggests words based on input.**
      Hint: Develop a function that takes a sequence of words and returns the most probable next word.
- [ ] **4. Create a CLI for users to interact with the suggestion engine.**  
      Hint: Use Python's `argparse` or `click` libraries to build an interactive command-line interface.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
python -m app.cli hello
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

The bundled `corpus/administrative_medical_sentences.txt` contains synthetic,
non-identifying examples covering scheduling, registration, documentation,
referrals, discharge, insurance, and billing. `app/corpus.py` lowercases text,
separates punctuation, normalizes whitespace, tokenizes sentences, and removes
empty or duplicate entries before the corpus is returned by `milestone_1()`.

`app/language_model.py` implements unigram, bigram, and trigram frequency
counters from tokenized sentences. `NGramLanguageModel.conditional_probability`
uses the corresponding zero-, one-, or two-token context and Laplace
add-one smoothing: `(count(context, word) + 1) / (count(context) + V)`,
where `V` is the observed vocabulary size. Unknown words and contexts receive
smoothed probabilities; an empty corpus has no vocabulary and returns `0.0`.
`milestone_2()` trains the model on the bundled corpus. The prediction model
does not call an external language model or AI API.

`app/prediction.py` uses observed trigram continuations first, then backs off
to bigrams and unigrams when the longer context has no observed continuations.
`suggest_next_words()` accepts either a sentence or token sequence and returns
up to the requested number of `Suggestion` values, each containing a word and
its conditional probability. Results are ordered by probability, with
alphabetical tie-breaking, and only words observed in the selected context
(or unigram corpus) are returned. `milestone_3()` loads the trained corpus
model by default and also accepts an existing model.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Allow users to specify different n-gram sizes for the language model.
- Implement a simple 'undo' feature for the last generated suggestion.
- Add a feature to save and load the trained language model to avoid retraining on every run.
