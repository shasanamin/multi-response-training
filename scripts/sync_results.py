from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from _bootstrap import PROJECT_ROOT  # noqa: F401

from config import load_config
from reporting import project_logs_root, rebuild_summary_index, write_project_run_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-config", default="configs/runtime/cluster.yaml")
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime_config = load_config(args.runtime_config)
    runs_root = Path(runtime_config["paths"]["runs_root"])

    summaries = sorted(runs_root.glob("*/summary.json"))
    if args.run_name:
        summaries = [path for path in summaries if path.parent.name == args.run_name]

    for summary_path in summaries:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        output_dir = write_project_run_artifacts(summary)
        print(f"Synced results for {summary['run_name']} -> {output_dir}")

    tmux_logs_dir = runs_root / "tmux_logs"
    project_logs_dir = project_logs_root()
    if tmux_logs_dir.exists():
        for log_path in sorted(tmux_logs_dir.glob("*.log")):
            shutil.copy2(log_path, project_logs_dir / log_path.name)
            print(f"Synced log {log_path.name}")

    index_path = rebuild_summary_index()
    print(f"Wrote summary index to {index_path}")


if __name__ == "__main__":
    main()
