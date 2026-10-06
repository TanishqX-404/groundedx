"""Domain-configured Jinja prompt rendering.

RAG and zero-shot prompts come from the same template; they differ only in the
evidence block and the one-line citation rule.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import Template

from groundedx.config import DomainConfig

EVIDENCE_CHAR_LIMIT = 1200
RAG_CITATION_RULE = (
    "Cite the IDs of the retrieved evidence you rely on; citations may only contain "
    "IDs listed under Retrieved evidence."
)
ZERO_SHOT_CITATION_RULE = "No evidence was retrieved, so citations must be an empty list."
NO_EVIDENCE = "(none)"


def render_evidence(retrieved: list[dict[str, Any]]) -> str:
    return "\n\n".join(
        f"[{item['chunk_id']}] score={float(item.get('score', 0.0)):.3f}\n"
        f"{item['text'][:EVIDENCE_CHAR_LIMIT]}"
        for item in retrieved
    )


def render_prompt(
    query: str,
    retrieved: list[dict[str, Any]],
    domain: DomainConfig,
    template_path: str | Path,
    *,
    taxonomy: list[str] | None = None,
    reference: str = "",
) -> str:
    """Render a prompt; an empty ``retrieved`` list gives the zero-shot prompt.

    ``taxonomy`` restricts the listed classes (hybrid candidates); ``reference``
    fills the optional fault-reference block of the docs/hybrid templates.
    """

    template = Template(Path(template_path).read_text(encoding="utf-8"))
    return template.render(
        domain=domain,
        taxonomy=taxonomy if taxonomy is not None else domain.classes,
        query=query,
        evidence=render_evidence(retrieved) if retrieved else NO_EVIDENCE,
        citation_rule=RAG_CITATION_RULE if retrieved else ZERO_SHOT_CITATION_RULE,
        reference=reference,
    )
