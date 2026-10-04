"""Domain-configured synthetic telemetry generation."""

from .kpi_generator import generate_dataset, simulate_window, split_dataset

__all__ = ["generate_dataset", "simulate_window", "split_dataset"]
