"""Terminal output: one shared rich console plus small formatting helpers."""

from __future__ import annotations

import math
import re
import sys
from collections.abc import Iterator
from contextlib import contextmanager

from rich.console import Console
from rich.markup import escape
from rich.theme import Theme

THEME = Theme(
    {
        "info": "cyan",
        "warn": "yellow",
        "error": "bold red",
        "good": "green",
        "muted": "dim",
        "accent": "bold magenta",
        "best": "bold green",
        "head": "bold",
    }
)

console = Console(theme=THEME, highlight=False)

# Rich redraws spinners and progress bars from a background thread. Browsers (Pyodide, where
# sys.platform is "emscripten") can't start threads, so there bars only redraw on updates.
LIVE_REFRESH = sys.platform != "emscripten"


def set_quiet(enabled: bool) -> None:
    console.quiet = enabled


@contextmanager
def quiet(enabled: bool = True) -> Iterator[None]:
    """Silence output inside the block (used by the Python API when verbose=False)."""
    previous = console.quiet
    console.quiet = previous or enabled
    try:
        yield
    finally:
        console.quiet = previous


def heading(text: str) -> None:
    console.print()
    console.rule(f"[head]{text}[/]", align="left", style="muted")


def info(text: str) -> None:
    console.print(f"[info]•[/] {text}")


def note(text: str) -> None:
    console.print(f"  [muted]{text}[/]")


def warn(text: str) -> None:
    console.print(f"[warn]![/] {text}")


def success(text: str) -> None:
    console.print(f"[good]✓[/] {text}")


def esc(value: object) -> str:
    """Escape text for rich markup: column names like 'width [cm]' must print as-is."""
    return escape(str(value))


def error(text: str, hint: str | None = None) -> None:
    """Print an error. Both parts are plain text (they often quote user data)."""
    console.print(f"[error]Error:[/] {esc(text)}", soft_wrap=True)
    if hint:
        console.print(f"[muted]{esc(hint)}[/]", soft_wrap=True)


def fmt_num(value: float | None, digits: int = 4) -> str:
    """Format a metric for display with about four significant digits.

    0.9312, 58.67, 1,234, 3.2e-05, or '—' for missing values.
    """
    if value is None:
        return "—"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(value):
        return "—"
    if math.isinf(value):
        return "∞" if value > 0 else "-∞"
    if value == 0:
        return "0"
    magnitude = abs(value)
    if magnitude >= 1e9 or magnitude < 1e-3:
        return f"{value:.3g}"
    if magnitude < 1:
        return f"{value:.{digits}f}"
    whole_digits = int(math.floor(math.log10(magnitude))) + 1
    return f"{value:,.{max(0, digits - whole_digits)}f}"


def fmt_value(value: object) -> str:
    """Format a data value (not a metric): whole numbers without decimals, others like fmt_num."""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)) or hasattr(value, "dtype"):
        try:
            number = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return str(value)
        if math.isfinite(number) and number.is_integer() and abs(number) < 1e15:
            return f"{int(number):,}"
        return fmt_num(number)
    return str(value)


def fmt_pct(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}%"


def fmt_duration(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes}m {secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


_DURATION = re.compile(r"(\d+(?:\.\d+)?)\s*(w|d|h|m|s)?", re.IGNORECASE)


def parse_duration(text: str | float | int | None) -> float | None:
    """Parse '90', '90s', '5m', '1h30m', '30d', '2w' into seconds. None stays None."""
    if text is None or text == "":
        return None
    if isinstance(text, (int, float)):
        return float(text)
    cleaned = str(text).strip().replace(" ", "")
    total = 0.0
    position = 0
    for match in _DURATION.finditer(cleaned):
        if match.start() != position:
            break
        amount, unit = float(match.group(1)), (match.group(2) or "s").lower()
        total += amount * {"w": 604800, "d": 86400, "h": 3600, "m": 60, "s": 1}[unit]
        position = match.end()
    if position != len(cleaned) or position == 0:
        from plainml.errors import PlainMLError

        raise PlainMLError(
            f"Couldn't understand the duration '{text}'.",
            hint="Use seconds or units, e.g. 90, 90s, 5m, 1h30m.",
        )
    return total
