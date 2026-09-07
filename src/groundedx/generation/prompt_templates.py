"""Domain-configured Jinja prompt rendering."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import Template

from groundedx.config import DomainConfig


def render_prompt(
    query: str, retrieved: list[dict[str, Any]], domain: DomainConfig, template_path: str | Path
) -> str:
    """Render a domain prompt using only the active taxonomy and evidence."""

    evidence = "\n\n".join(
        f"[{item['chunk_id']}] doc_type={item.get('doc_type', 'evidence')}\n{item['text']}"
        for item in retrieved
    )
    template = Template(Path(template_path).read_text(encoding="utf-8"))
    return template.render(domain=domain, taxonomy=domain.classes, query=query, evidence=evidence)
