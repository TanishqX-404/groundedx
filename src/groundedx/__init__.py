"""Grounded diagnosis from telemetry, alerts, and logs."""

from .config import DomainConfig, load_domain_config
from .generation.validation import Diagnosis, validate_diagnosis

__all__ = ["Diagnosis", "DomainConfig", "load_domain_config", "validate_diagnosis"]
__version__ = "0.1.0"
