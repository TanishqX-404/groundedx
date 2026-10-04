"""Optional llama.cpp generation adapter (GBNF-constrained, greedy decoding)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from groundedx.generation.grammar import build_gbnf_grammar
from groundedx.generation.validation import InvalidOutput, parse_and_validate

TEMPERATURE = 0.0
MAX_TOKENS = 384


class LlamaCppClient:
    """Lazy-loading local GGUF client. Every call passes a real ``LlamaGrammar``."""

    def __init__(
        self, model_path: str | Path, *, n_ctx: int = 4096, n_gpu_layers: int = -1, seed: int = 0
    ) -> None:
        try:
            from llama_cpp import Llama, LlamaGrammar
        except ImportError as exc:
            raise RuntimeError("Install groundedx[llama] to use local GGUF generation") from exc
        self._grammar_cls = LlamaGrammar
        self._grammar_cache: dict[tuple, Any] = {}
        self.model_path = Path(model_path)
        self._llm = Llama(
            model_path=str(model_path),
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            seed=seed,
            verbose=False,
        )

    def grammar(self, classes: list[str], citation_ids: list[str]) -> Any:
        key = (tuple(classes), tuple(citation_ids))
        if key not in self._grammar_cache:
            if len(self._grammar_cache) > 4096:
                self._grammar_cache.clear()
            self._grammar_cache[key] = self._grammar_cls.from_string(
                build_gbnf_grammar(classes, citation_ids), verbose=False
            )
        return self._grammar_cache[key]

    def complete(self, prompt: str, classes: list[str], citation_ids: list[str]) -> dict[str, Any]:
        """Run one constrained generation and return raw output plus timing."""

        grammar = self.grammar(classes, citation_ids)
        start = time.perf_counter()
        output = self._llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            grammar=grammar,
        )
        latency_ms = (time.perf_counter() - start) * 1000.0
        choice = output["choices"][0]
        usage = output.get("usage", {})
        return {
            "raw": choice["message"]["content"] or "",
            "finish_reason": choice.get("finish_reason"),
            "latency_ms": latency_ms,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
        }

    def diagnose(self, prompt: str, classes: list[str], citation_ids: list[str]) -> dict[str, Any]:
        """Generate and validate. Invalid outputs are reported, never replaced."""

        out = self.complete(prompt, classes, citation_ids)
        try:
            diagnosis = parse_and_validate(
                out["raw"], set(classes), set(citation_ids), out["finish_reason"]
            )
            out.update(valid=True, invalid_reason=None, diagnosis=diagnosis.model_dump())
        except InvalidOutput as exc:
            out.update(valid=False, invalid_reason=exc.reason, diagnosis=None)
        return out
