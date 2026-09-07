"""Thin CLI wrapper for the generic simulator."""

from __future__ import annotations

import argparse

from groundedx.config import load_domain_bundle
from groundedx.simulator.kpi_generator import generate_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default="domains/oran")
    parser.add_argument("--n", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    rows = generate_dataset(load_domain_bundle(args.domain), args.n, args.seed)
    print(f"Generated {len(rows)} windows for {args.domain}")


if __name__ == "__main__":
    main()

