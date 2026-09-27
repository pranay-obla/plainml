"""Regenerate docs/commands.md from the CLI's own --help text:  python docs/generate_commands.py"""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from plainml.cli import SECTIONS, cli


def help_for(args: list[str]) -> str:
    result = CliRunner().invoke(cli, [*args, "--help"], prog_name="plainml", terminal_width=100)
    if result.exit_code != 0:
        raise SystemExit(result.output)
    return result.output.rstrip()


def main() -> None:
    lines = [
        "# Command reference",
        "",
        "Generated from `plainml COMMAND --help` (run `python docs/generate_commands.py` to refresh).",
        "",
        "## plainml",
        "",
        "```text",
        help_for([]),
        "```",
        "",
    ]
    for title, names in SECTIONS:
        lines += [f"## {title}", ""]
        for name in names:
            lines += [f"### plainml {name}", "", "```text", help_for([name]), "```", ""]
    target = Path(__file__).with_name("commands.md")
    target.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
