import sys


def render_plan_mode_toggle(enabled: bool) -> None:
    """Print plan mode status after toggling."""
    print(f"Plan mode {'on' if enabled else 'off'}")


def render_goodbye() -> None:
    """Print exit message."""
    print("\nGoodbye!")


def render_error(error: Exception) -> None:
    """Print an error message to stderr."""
    print(f"\nError: {error}", file=sys.stderr)


def render_final_response(response: str) -> None:
    """Print a one-shot task response."""
    print("\n--- FINAL RESPONSE ---")
    print(response)
