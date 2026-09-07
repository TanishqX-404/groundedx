"""Generate a compact GBNF grammar for the active taxonomy."""

from __future__ import annotations

import json


def _quoted(value: str) -> str:
    return json.dumps(value, ensure_ascii=True)


def build_gbnf_grammar(classes: list[str], citation_ids: list[str]) -> str:
    """Return a llama.cpp-compatible grammar with closed root/citation values."""

    if not classes:
        raise ValueError("At least one taxonomy class is required")
    class_rule = " | ".join(_quoted(item) for item in classes)
    citation_rule = '""' if not citation_ids else " | ".join(_quoted(item) for item in citation_ids)
    return f"""root ::= "{{" ws "\\\"root_cause\\\":" ws class ws "," ws "\\\"confidence\\\":" ws number ws "," ws "\\\"remediation\\\":" ws string ws "," ws "\\\"explanation\\\":" ws string ws "," ws "\\\"citations\\\":" ws citations ws "}}"
class ::= {class_rule}
citations ::= "[" ws ("" | citation (ws "," ws citation)*) ws "]"
citation ::= {citation_rule}
string ::= "\\\"" ([^"\\] | "\\\\" .)* "\\\""
number ::= [0-1] [0-9]* ("." [0-9]+)?
ws ::= [ \t\n]*
"""
