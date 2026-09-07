"""Optional llama.cpp generation adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from groundedx.config import DomainConfig
from groundedx.generation.grammar import build_gbnf_grammar
from groundedx.generation.validation import Diagnosis, validate_diagnosis


class LlamaCppClient:
    """Lazy-loading local GGUF client with post-generation validation."""

    def __init__(self, model_path: str | Path, *, n_ctx: int = 4096, n_gpu_layers: int = 0) -> None:
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise RuntimeError("Install groundedx[llama] to use local GGUF generation") from exc
        self._llm = Llama(
            model_path=str(model_path), n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, verbose=False
        )

    def generate(self, prompt: str, domain: DomainConfig, retrieved_ids: set[str]) -> Diagnosis:
        """Generate and reject outputs outside the active closed schema."""

        grammar = build_gbnf_grammar(domain.classes, sorted(retrieved_ids))
        output = self._llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=384,
            grammar=grammar,
        )
        payload: dict[str, Any] = json.loads(output["choices"][0]["message"]["content"])
        return validate_diagnosis(payload, set(domain.classes), retrieved_ids)
