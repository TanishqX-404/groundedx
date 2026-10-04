"""Generate a GBNF grammar for the active taxonomy and retrieved evidence.

The grammar fixes the JSON key order and restricts ``root_cause`` to the closed
taxonomy and ``citations`` to the IDs retrieved for this query. With no
retrieved IDs (zero-shot) the only admissible citation list is ``[]``.
"""

from __future__ import annotations

MAX_TEXT_CHARS = 240
MAX_CITATIONS = 5


def _literal(value: str) -> str:
    """A GBNF literal that matches the JSON string ``"value"`` (quotes included)."""

    if any(ch in value for ch in '"\\\n'):
        raise ValueError(f"Unsupported character in grammar literal: {value!r}")
    return f'"\\"{value}\\""'


def build_gbnf_grammar(classes: list[str], citation_ids: list[str]) -> str:
    """Return a llama.cpp GBNF grammar for one diagnosis call."""

    if not classes:
        raise ValueError("At least one taxonomy class is required")
    cause = " | ".join(_literal(item) for item in classes)
    if citation_ids:
        cite = " | ".join(_literal(item) for item in citation_ids)
        cites = (
            f'"[" ws "]" | "[" ws cite (ws "," ws cite){{0,{MAX_CITATIONS - 1}}} ws "]"\n'
            f"cite ::= {cite}"
        )
    else:
        cites = '"[" ws "]"'
    return (
        'root ::= "{" ws "\\"root_cause\\":" ws cause ws "," ws '
        '"\\"confidence\\":" ws conf ws "," ws '
        '"\\"remediation\\":" ws text ws "," ws '
        '"\\"explanation\\":" ws text ws "," ws '
        '"\\"citations\\":" ws cites ws "}"\n'
        f"cause ::= {cause}\n"
        'conf ::= "0." [0-9] [0-9]? | "1.0"\n'
        f'text ::= "\\"" char{{1,{MAX_TEXT_CHARS}}} "\\""\n'
        'char ::= [^"\\\\\\x00-\\x1F] | "\\\\" ["\\\\/bfnrt]\n'
        f"cites ::= {cites}\n"
        "ws ::= [ \\n]{0,2}\n"
    )
