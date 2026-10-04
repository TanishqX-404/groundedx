"""llama.cpp ``llama-server`` backend (GBNF-constrained, greedy decoding).

Same interface as ``LlamaCppClient``. Preferred for the revision runs because
llama.cpp's native sampler checks the grammar only for the sampled token and
falls back to full-vocabulary masking when that token is rejected. Under greedy
decoding this yields exactly the grammar-constrained argmax, at a fraction of
the per-token cost of llama-cpp-python's full-vocabulary grammar pass.
"""

from __future__ import annotations

import json
import socket
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any

from groundedx.generation.grammar import build_gbnf_grammar
from groundedx.generation.validation import InvalidOutput, parse_and_validate

TEMPERATURE = 0.0
MAX_TOKENS = 384


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class LlamaServerClient:
    """Starts ``llama-server`` for one GGUF model and talks to it over HTTP."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        server_bin: str | Path,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
        seed: int = 0,
        log_path: str | Path | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.port = _free_port()
        self.base = f"http://127.0.0.1:{self.port}"
        ngl = "999" if n_gpu_layers < 0 else str(n_gpu_layers)
        cmd = [
            str(Path(server_bin).resolve()),
            "-m",
            str(Path(model_path).resolve()),
            "-c",
            str(n_ctx),
            "-ngl",
            ngl,
            "--port",
            str(self.port),
            "--host",
            "127.0.0.1",
            "-np",
            "1",
            "--seed",
            str(seed),
            "--no-webui",
        ]
        self._log = open(log_path, "a", encoding="utf-8") if log_path else subprocess.DEVNULL
        self._proc = subprocess.Popen(cmd, stdout=self._log, stderr=subprocess.STDOUT)
        self._wait_ready()
        self.server_version = self._get("/props").get("build_info")

    def _get(self, path: str) -> dict[str, Any]:
        with urllib.request.urlopen(self.base + path, timeout=30) as response:
            return json.loads(response.read())

    def _wait_ready(self, timeout: float = 600.0) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(f"llama-server exited with code {self._proc.returncode}")
            try:
                if self._get("/health").get("status") == "ok":
                    return
            except Exception:
                pass
            time.sleep(0.5)
        raise TimeoutError("llama-server did not become ready")

    def close(self) -> None:
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._proc.kill()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def complete(self, prompt: str, classes: list[str], citation_ids: list[str]) -> dict[str, Any]:
        body = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": TEMPERATURE,
            "top_k": 1,
            "max_tokens": MAX_TOKENS,
            "grammar": build_gbnf_grammar(classes, citation_ids),
            "cache_prompt": True,
        }
        request = urllib.request.Request(
            self.base + "/v1/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        start = time.perf_counter()
        with urllib.request.urlopen(request, timeout=600) as response:
            output = json.loads(response.read())
        latency_ms = (time.perf_counter() - start) * 1000.0
        choice = output["choices"][0]
        usage = output.get("usage", {})
        return {
            "raw": choice["message"].get("content") or "",
            "finish_reason": choice.get("finish_reason"),
            "latency_ms": latency_ms,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "server_timings": output.get("timings"),
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
