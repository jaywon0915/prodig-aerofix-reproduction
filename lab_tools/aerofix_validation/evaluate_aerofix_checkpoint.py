#!/usr/bin/env python3
"""Evaluate a trained aeroFix .pkl checkpoint against a prepared data.json.

Example:
    python scripts/evaluate_aerofix_checkpoint.py \
        --config aerofix_training/configs/citysample_1_graphdeco.yaml \
        --checkpoint /path/to/checkpoints/aerofix_outputs/aerofix_train_a6000_3gpu/checkpoints/model_10000.pkl \
        --data /path/to/matrixcity_eval/data.json \
        --output /path/to/matrixcity_eval/eval_output_rerun
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from anchor_prodig.aerofix_factory import evaluate_aerofix, load_factory_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an aeroFix diffusion checkpoint")
    parser.add_argument("--config", type=Path, required=True, help="Any aerofix_training config yaml (only used for structural defaults)")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to model_*.pkl")
    parser.add_argument("--data", type=Path, required=True, help="Path to data.json")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for images/ and metrics.json")
    parser.add_argument("--dreamsim", action="store_true", help="Enable DreamSim metric (fails on peft version conflicts, off by default)")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    cfg = load_factory_config(args.config)
    cfg.evaluation.backend = "aerofix"
    cfg.evaluation.checkpoint_format = "auto"
    cfg.evaluation.compute_dreamsim = args.dreamsim
    cfg.evaluation.device = args.device

    output_dir = evaluate_aerofix(cfg, args.checkpoint, args.data, args.output)

    metrics_path = Path(output_dir) / "metrics.json"
    print(f"\nDone. Images + metrics.json written to: {output_dir}")
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        print(json.dumps(metrics.get("mean", metrics), indent=2))


if __name__ == "__main__":
    main()
