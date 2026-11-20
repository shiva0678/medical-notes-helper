"""Tests for the interactive command-line application."""

from unittest.mock import Mock

import pytest

from app.cli import main, run_interactive
from app.language_model import NGramLanguageModel


def _cli_model() -> NGramLanguageModel:
    return NGramLanguageModel().fit(
        [
            ["the", "patient", "was", "advised", "to", "call"],
            ["the", "patient", "was", "advised", "to", "return"],
            ["the", "patient", "was", "scheduled", "for", "review"],
            ["billing", "team", "reviewed", "form"],
        ]
    )


def _provide_inputs(monkeypatch, responses: list[str]) -> None:
    monkeypatch.setattr("builtins.input", Mock(side_effect=responses))


def test_cli_greets_displays_suggestions_appends_and_regenerates(
    monkeypatch, capsys
):
    model_loader = Mock(return_value=_cli_model())
    monkeypatch.setattr("app.cli.milestone_2", model_loader)
    _provide_inputs(monkeypatch, ["The patient was", "1", "exit"])

    run_interactive()

    output = capsys.readouterr().out
    assert "Medical Notes Helper" in output
    assert "Welcome!" in output
    assert "Suggested next words:" in output
    assert "1. advised (Probability:" in output
    assert "Updated sentence: The patient was advised" in output
    assert "1. to (Probability:" in output
    assert output.count("Suggested next words:") == 2
    assert "Thank you for using Medical Notes Helper!" in output
    model_loader.assert_called_once_with()


def test_cli_accepts_new_text_and_handles_invalid_selection(monkeypatch, capsys):
    monkeypatch.setattr("app.cli.milestone_2", _cli_model)
    _provide_inputs(
        monkeypatch,
        ["a new sentence", "99", "billing team", "exit"],
    )

    run_interactive()

    output = capsys.readouterr().out
    assert "Invalid selection." in output
    assert "Using new text: billing team" in output
    assert "reviewed (Probability:" in output


def test_cli_handles_empty_text_and_empty_choice(monkeypatch, capsys):
    monkeypatch.setattr("app.cli.milestone_2", _cli_model)
    _provide_inputs(monkeypatch, ["", "", "exit"])

    run_interactive()

    output = capsys.readouterr().out
    assert "Suggested next words:" in output
    assert "Please enter new text, choose a suggestion, or type 'exit'." in output
    assert "Thank you for using Medical Notes Helper!" in output


def test_cli_reports_when_model_has_no_suggestions(monkeypatch, capsys):
    monkeypatch.setattr(
        "app.cli.milestone_2", lambda: NGramLanguageModel().fit([])
    )
    _provide_inputs(monkeypatch, ["unknown context", "exit"])

    run_interactive()

    output = capsys.readouterr().out
    assert "No suggestions are available for this text." in output


@pytest.mark.parametrize("exception", [EOFError, KeyboardInterrupt])
def test_cli_exits_gracefully_on_eof_or_interrupt(
    monkeypatch, capsys, exception
):
    monkeypatch.setattr("app.cli.milestone_2", _cli_model)
    monkeypatch.setattr("builtins.input", Mock(side_effect=exception))

    run_interactive()

    assert "Thank you for using Medical Notes Helper!" in capsys.readouterr().out


def test_main_supports_top_five_and_preserves_hello_command(monkeypatch, capsys):
    interactive = Mock()
    monkeypatch.setattr("app.cli.run_interactive", interactive)

    main(["--top-k", "5"])
    interactive.assert_called_once_with(top_k=5)

    main(["hello"])
    assert "is ready" in capsys.readouterr().out
