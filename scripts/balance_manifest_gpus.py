from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from _bootstrap import PROJECT_ROOT  # noqa: F401


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--gpu-ids", required=True, help="Comma-separated GPU ids, e.g. 0,1")
    parser.add_argument("--output", required=True)
    parser.add_argument("--incomplete-only", action="store_true")
    return parser.parse_args()


def _load_manifest(path: Path) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return payload, [dict(experiment) for experiment in payload.get("experiments", [])]
    if isinstance(payload, list):
        return None, [dict(experiment) for experiment in payload]
    raise TypeError(f"Unsupported manifest type: {type(payload)!r}")


def _summary_exists(run_name: str) -> bool:
    return (PROJECT_ROOT / "results" / "runs" / run_name / "summary.json").exists()


def _parse_gpu_ids(raw: str) -> list[int]:
    values = [int(chunk.strip()) for chunk in raw.split(",") if chunk.strip()]
    if not values:
        raise ValueError("At least one GPU id is required.")
    return values


def main() -> None:
    args = parse_args()
    gpu_ids = _parse_gpu_ids(args.gpu_ids)
    payload, experiments = _load_manifest(PROJECT_ROOT / args.manifest)

    if args.incomplete_only:
        experiments = [experiment for experiment in experiments if not _summary_exists(experiment["run_name"])]

    loads = {gpu_id: 0.0 for gpu_id in gpu_ids}
    queue_orders = {gpu_id: 1 for gpu_id in gpu_ids}
    assigned: list[dict[str, Any]] = []

    sorted_experiments = sorted(
        experiments,
        key=lambda experiment: (
            -float(experiment.get("estimated_cost", 1.0)),
            int(experiment.get("queue_order", 0)),
            str(experiment["run_name"]),
        ),
    )

    for experiment in sorted_experiments:
        gpu_id = min(gpu_ids, key=lambda candidate: (loads[candidate], queue_orders[candidate], candidate))
        assigned_experiment = dict(experiment)
        assigned_experiment["gpu_id"] = int(gpu_id)
        assigned_experiment["queue_order"] = int(queue_orders[gpu_id])
        queue_orders[gpu_id] += 1
        loads[gpu_id] += float(experiment.get("estimated_cost", 1.0))
        assigned.append(assigned_experiment)

    assigned.sort(key=lambda experiment: (int(experiment["gpu_id"]), int(experiment["queue_order"])))

    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if payload is not None:
        output_payload = dict(payload)
        output_payload["experiments"] = assigned
    else:
        output_payload = assigned

    output_path.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")
    print(f"Wrote {len(assigned)} experiments to {output_path}")
    for gpu_id in gpu_ids:
        gpu_runs = [experiment["run_name"] for experiment in assigned if int(experiment["gpu_id"]) == gpu_id]
        print(f"GPU {gpu_id}: {len(gpu_runs)} runs, estimated load {loads[gpu_id]:.3f}")


if __name__ == "__main__":
    main()
