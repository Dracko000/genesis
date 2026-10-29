"""Run a Genesis experiment and write a reproducible JSON result.

Examples
--------
    py scripts/run_experiment.py E001
    py scripts/run_experiment.py E001 --seeds 0,1,2 --epochs 300
    py scripts/run_experiment.py HYBRID --config configs/hybrid.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from genesis.evaluation.experiments import (  # noqa: E402
    E001Config,
    E002Config,
    E003Config,
    HybridAblationConfig,
    run_e001,
    run_e002,
    run_e003,
    run_hybrid_ablation,
    save_result,
)

REGISTRY = {
    "E001": (E001Config, run_e001, "E001_emergent_structure"),
    "E002": (E002Config, run_e002, "E002_structure_reuse"),
    "E003": (E003Config, run_e003, "E003_representation_revision"),
    "HYBRID": (HybridAblationConfig, run_hybrid_ablation, "E004_hybrid_ablation"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Genesis experiment")
    parser.add_argument("experiment", choices=sorted(REGISTRY))
    parser.add_argument("--config", type=Path, help="JSON file with config overrides")
    parser.add_argument("--seeds", help="comma-separated integer seeds")
    parser.add_argument("--epochs", type=int, help="override neural epochs")
    parser.add_argument("--output", type=Path, help="explicit output JSON path")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments",
        help="base directory for generated result files",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_cls, runner, directory = REGISTRY[args.experiment]

    overrides: dict[str, object] = {}
    if args.config:
        overrides.update(json.loads(args.config.read_text(encoding="utf-8")))
    if args.seeds:
        overrides["seeds"] = [int(value) for value in args.seeds.split(",") if value.strip()]
    if args.epochs:
        neural = dict(overrides.get("neural", {}))  # type: ignore[arg-type]
        neural["epochs"] = args.epochs
        overrides["neural"] = neural

    config = config_cls.model_validate(overrides)
    result = runner(config)

    if args.output:
        output = args.output
    else:
        seed_tag = "-".join(str(seed) for seed in getattr(config, "seeds", []))
        output = args.output_dir / directory / f"results_seeds_{seed_tag or 'default'}.json"
    save_result(result, output)

    print(json.dumps(result["criterion"], indent=2))
    print(f"result written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
