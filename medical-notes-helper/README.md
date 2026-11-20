[![Built with askThili](https://msrit.ask.thili.ai/api/v1/public/projects/ckmxkb47xb/badge.svg)](https://msrit.ask.thili.ai/b/ckmxkb47xb)
# Medical Notes Helper

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/ckmxkb47xb/badge.svg)](https://ask.thili.ai/b/ckmxkb47xb)

Medical Notes Helper is an educational command-line tool that helps clinic
staff draft administrative note text by suggesting likely next words. It uses
a small n-gram language model trained only on bundled synthetic examples.
It is not a clinical decision tool. Do not enter real patient records or
personally identifiable health information.

## Features

- Synthetic administrative corpus for scheduling, registration,
  documentation, referrals, discharge, insurance, and billing.
- In-project unigram, bigram, and trigram counting with Laplace smoothing.
- Context-aware top-three or top-five word suggestions with probabilities.
- Interactive sentence editing: accept a suggestion or enter new text.
- Deterministic ranking and graceful handling of empty inputs, invalid choices,
  Ctrl+C, and end-of-file.
- No external language model or AI API is used for prediction.

## Technologies

- Python 3.12 (the code also runs on Python 3.13).
- Python standard library for the CLI and n-gram model.
- `pytest` for tests; `httpx` remains a dependency for the optional `app.llm`
  helper and is not used by the prediction engine.

## Project structure

```text
repository-root/
├── .github/
│   └── workflows/ci.yml
└── medical-notes-helper/
    ├── app/
    │   ├── __init__.py
    │   ├── cli.py
    │   ├── core.py
    │   ├── corpus.py
    │   ├── language_model.py
    │   ├── prediction.py
    │   ├── llm.py
    │   └── storage.py
    ├── corpus/
    │   └── administrative_medical_sentences.txt
    ├── tests/
    │   ├── test_cli.py
    │   ├── test_core.py
    │   └── test_smoke.py
    ├── .env.example
    ├── .gitignore
    ├── pytest.ini
    └── requirements.txt
```

## Installation

Install Python 3.12 and Git, then clone the repository:

```powershell
git clone https://github.com/shiva0678/medical-notes-helper.git
Set-Location .\medical-notes-helper\medical-notes-helper
```

Create and activate a virtual environment, then install dependencies:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On macOS or Linux, use the corresponding commands:

```bash
git clone https://github.com/shiva0678/medical-notes-helper.git
cd medical-notes-helper/medical-notes-helper
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The local n-gram CLI does not require a `.env` file or API credentials.
`.env.example` documents optional settings for the separate `app.llm` helper.
The `.env` file is ignored by Git.

## Run the CLI

From the project directory, start the interactive helper (three suggestions
are shown by default):

```bash
python -m app.cli
```

To show five suggestions, or to explicitly start interactive mode:

```bash
python -m app.cli --top-k 5
python -m app.cli run
```

The existing setup check is also available:

```bash
python -m app.cli hello
```

Example interaction (suggestions depend on the bundled corpus):

```text
Medical Notes Helper
Welcome! Enter a sentence to get next-word suggestions.

Enter your text: the patient

Suggested next words:
1. requested (Probability: 0.013)
2. selected (Probability: 0.013)

Choose a suggestion (1-2), enter new text, or type 'exit': 1

Updated sentence: the patient requested

Suggested next words:
1. an (Probability: 0.013)

Choose a suggestion (1-1), enter new text, or type 'exit': exit
Thank you for using Medical Notes Helper!
```

Type a suggestion number to append that word and refresh suggestions. Enter
different text to replace the current sentence, or type `exit` to quit. An
empty sentence uses unigram suggestions; if the model has no available
continuations, the CLI explains that and lets you enter new text. Ctrl+C and
Ctrl+D/EOF end the session with a farewell message.

## How the model works

`app/corpus.py` loads and cleans the synthetic sentences. The
`NGramLanguageModel` in `app/language_model.py` counts individual words,
two-word sequences, and three-word sequences. The prediction engine in
`app/prediction.py` selects observed continuations for the longest available
context: trigram first, then bigram, then unigram. It ranks candidates by their
conditional probability and returns only words observed in that context (or
in the corpus for unigram fallback).

For a word `w` and context `c`, the model applies Laplace add-one smoothing:

```text
P(w | c) = (count(c, w) + 1) / (count(c) + V)
```

Here, `V` is the number of unique words in the training corpus. Adding one to
each count avoids zero probabilities for unseen words and contexts. Smoothing
is used to score observed candidates; it does not add arbitrary unseen words
to the suggestion list.

## Tests

Run the complete test suite from the project directory:

```bash
python -m pytest -q
```

GitHub Actions runs the test suite with Python 3.12 on pushes and pull requests.

## GitHub

Repository: <https://github.com/shiva0678/medical-notes-helper>

## Project milestones

- [x] Curate a synthetic administrative corpus.
- [x] Implement the unigram, bigram, and trigram model with smoothing.
- [x] Develop context-aware next-word prediction.
- [x] Build the interactive CLI.
