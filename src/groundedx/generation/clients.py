"""Pick a llama.cpp backend for the run scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def make_client(
    backend: str,
    model_path: str | Path,
    *,
    n_ctx: int,
    n_gpu_layers: int,
    server_bin: str | None = None,
    log_path: str | Path | None = None,
) -> Any:
    if backend == "server":
        from groundedx.generation.llama_server_client import LlamaServerClient

        if not server_bin:
            raise ValueError("--server-bin is required for the server backend")
        return LlamaServerClient(
            model_path,
            server_bin=server_bin,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            log_path=log_path,
        )
    from groundedx.generation.llama_cpp_client import LlamaCppClient

    return LlamaCppClient(model_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers)


def backend_info(client: Any) -> dict[str, Any]:
    return {
        "backend": type(client).__name__,
        "llama_server_build": getattr(client, "server_version", None),
    }
