"""Escape untrusted text before it reaches a terminal."""

from __future__ import annotations

_NAMED_ESCAPES = {"\t": "\\t", "\n": "\\n", "\r": "\\r"}
"""Short escapes for common whitespace controls, as in Python literals."""


def _escape_character(character: str) -> str:
    r"""Return a Python-style backslash escape for one character.

    Args:
        character: Single character to escape.

    Returns:
        `\t`, `\n`, `\r`, `\xHH`, `\uHHHH`, or `\UHHHHHHHH` escape
        sequence.
    """
    named = _NAMED_ESCAPES.get(character)
    if named is not None:
        return named
    code = ord(character)
    if code <= 0xFF:
        return f"\\x{code:02x}"
    if code <= 0xFFFF:
        return f"\\u{code:04x}"
    return f"\\U{code:08x}"


def escape_text(text: str) -> str:
    """Escape every character that `str.isprintable` rejects.

    Store names, metadata, and file names can contain characters that a
    terminal interprets, such as escape sequences that move the cursor or
    rewrite earlier output. This function replaces C0 and C1 control
    characters, DEL, line and paragraph separators, bidirectional formatting
    characters, and lone surrogates from undecodable file names with
    backslash escapes. Other characters, including printable non-ASCII text,
    pass through unchanged. Backslashes are not escaped, so the result is
    for display only.

    Args:
        text: Untrusted text.

    Returns:
        Text containing only printable characters.
    """
    if text.isprintable():
        return text
    return "".join(
        character if character.isprintable() else _escape_character(character)
        for character in text
    )


def escape_lines(text: str) -> str:
    """Escape each line of multiline text and keep the line feeds.

    Args:
        text: Untrusted multiline text.

    Returns:
        Text whose lines contain only printable characters.
    """
    return "\n".join(escape_text(line) for line in text.split("\n"))


__all__ = ["escape_lines", "escape_text"]
