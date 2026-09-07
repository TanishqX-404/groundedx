"""Runtime-loaded domain configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class FaultSpec(BaseModel):
    """One closed-taxonomy diagnosis class and its synthetic signature."""

    name: str = Field(alias="class")
    domain: str = "system"
    alarms: list[str] = []
    kpi_effects: dict[str, float] = {}
    pattern: str = "step"
    remediation: str = "Review the relevant operations runbook."

    model_config = {"populate_by_name": True}


class MetricSpec(BaseModel):
    """Metric baseline and optional display metadata."""

    name: str
    baseline: float = 0.0
    noise_fraction: float = 0.13
    unit: str = ""


class DomainConfig(BaseModel):
    """All domain data needed by the generic pipeline."""

    name: str
    taxonomy: list[FaultSpec]
    metrics: list[MetricSpec]
    alarm_severity_distribution: dict[str, float] = {
        "minor": 0.15,
        "major": 0.55,
        "critical": 0.30,
    }
    distractor_probability: float = 0.70
    true_alarm_probability: float = 0.42
    extra_alarm_probability: float = 0.80
    log_templates: list[str] = []
    max_log_lines: int = 4
    retrieval: dict[str, Any] = {}

    @property
    def classes(self) -> list[str]:
        return [fault.name for fault in self.taxonomy]

    @property
    def metric_names(self) -> list[str]:
        return [metric.name for metric in self.metrics]

    def fault(self, name: str) -> FaultSpec:
        for item in self.taxonomy:
            if item.name == name:
                return item
        raise KeyError(f"Unknown taxonomy class: {name}")


def load_domain_config(path: str | Path) -> DomainConfig:
    """Load and validate a YAML domain configuration."""

    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    payload["name"] = payload.get("name", config_path.parent.name)
    return DomainConfig.model_validate(payload)


def load_domain_bundle(directory: str | Path) -> DomainConfig:
    """Load ``taxonomy.yaml`` and ``kpi_schema.yaml`` from one domain directory."""

    domain_dir = Path(directory)
    taxonomy_payload = (
        yaml.safe_load((domain_dir / "taxonomy.yaml").read_text(encoding="utf-8")) or {}
    )
    kpi_payload = yaml.safe_load((domain_dir / "kpi_schema.yaml").read_text(encoding="utf-8")) or {}
    merged = {**kpi_payload, **taxonomy_payload}
    merged["name"] = taxonomy_payload.get("name", domain_dir.name)
    return DomainConfig.model_validate(merged)
