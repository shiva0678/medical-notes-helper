"""Interactive command-line interface for Medical Notes Helper."""

import argparse
import sys

from app.core import milestone_2
from app.prediction import Suggestion, suggest_next_words


def _show_suggestions(suggestions: list[Suggestion]) -> None:
    if not suggestions:
        print("\nNo suggestions are available for this text.")
        return

    print("\nSuggested next words:")
    for index, suggestion in enumerate(suggestions, start=1):
        print(f"{index}. {suggestion.word} (Probability: {suggestion.probability:.3f})")


def run_interactive(top_k: int = 3) -> None:
    """Run an interactive drafting session using one trained model."""
    print("Medical Notes Helper")
    print("Welcome! Enter a sentence to get next-word suggestions.")

    try:
        model = milestone_2()
    except OSError as error:
        print(f"Unable to load the synthetic corpus: {error}", file=sys.stderr)
        return

    try:
        current_text = input("\nEnter your text: ").strip()
        while True:
            if current_text.lower() == "exit":
                break

            suggestions = suggest_next_words(model, current_text, top_k=top_k)
            _show_suggestions(suggestions)

            if suggestions:
                prompt = (
                    f"\nChoose a suggestion (1-{len(suggestions)}), "
                    "enter new text, or type 'exit': "
                )
            else:
                prompt = "\nEnter new text or type 'exit': "

            response = input(prompt).strip()
            if response.lower() == "exit":
                break
            if not response:
                print("Please enter new text, choose a suggestion, or type 'exit'.")
                continue

            try:
                selection = int(response)
            except ValueError:
                current_text = response
                print(f"\nUsing new text: {current_text}")
                continue

            if selection < 1 or selection > len(suggestions):
                if suggestions:
                    print(
                        f"Invalid selection. Enter a number from 1 to "
                        f"{len(suggestions)}, new text, or 'exit'."
                    )
                else:
                    print(
                        "No numbered suggestions are available. "
                        "Enter new text or 'exit'."
                    )
                continue

            selected = suggestions[selection - 1]
            current_text = f"{current_text.rstrip()} {selected.word}".strip()
            print(f"\nUpdated sentence: {current_text}")
    except (EOFError, KeyboardInterrupt):
        print()

    print("Thank you for using Medical Notes Helper!")


def main(argv: list[str] | None = None) -> None:
    """Parse CLI options and start the interactive app or the wiring check."""
    parser = argparse.ArgumentParser(
        prog="medical-notes-helper",
        description="Draft administrative notes with local n-gram suggestions.",
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=("run", "hello"),
        default="run",
        help="run the interactive helper (default) or check that it is wired up",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        choices=(3, 5),
        default=3,
        help="number of suggestions to show (default: 3)",
    )
    args = parser.parse_args(argv)

    if args.command == "hello":
        print("Medical Notes Helper is ready. Open README.md for project details.")
        return

    run_interactive(top_k=args.top_k)


if __name__ == "__main__":
    main()
