from __future__ import annotations

import argparse
import subprocess
import sys
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-ids", required=True, help="Comma-separated GPU ids to monitor, e.g. 0,1.")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--min-idle-seconds", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=int, default=0)
    parser.add_argument("--memory-threshold-mib", type=int, default=0)
    parser.add_argument("--util-threshold", type=int, default=0)
    return parser.parse_args()


def _parse_gpu_ids(raw: str) -> list[int]:
    values = [int(chunk.strip()) for chunk in raw.split(",") if chunk.strip()]
    if not values:
        raise ValueError("At least one GPU id must be provided.")
    return values


def _query_gpu_state() -> dict[int, tuple[int, int]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    states: dict[int, tuple[int, int]] = {}
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        states[int(parts[0])] = (int(parts[1]), int(parts[2]))
    return states


def main() -> None:
    args = parse_args()
    gpu_ids = _parse_gpu_ids(args.gpu_ids)
    idle_since: dict[int, float] = {}
    deadline = time.monotonic() + args.timeout_seconds if args.timeout_seconds > 0 else None

    while True:
        now = time.monotonic()
        summaries: list[str] = []
        try:
            states = _query_gpu_state()
        except subprocess.CalledProcessError as exc:
            states = {}
            message = (exc.stderr or exc.stdout or str(exc)).strip()
            summaries.append(f"nvidia-smi-error:{message or exc.returncode}")

        for gpu_id in gpu_ids:
            memory_used, utilization = states.get(gpu_id, (10**9, 10**9))
            summaries.append(f"{gpu_id}:{memory_used}MiB/{utilization}%")
            if memory_used <= args.memory_threshold_mib and utilization <= args.util_threshold:
                idle_since.setdefault(gpu_id, now)
                if now - idle_since[gpu_id] >= args.min_idle_seconds:
                    print(gpu_id)
                    return
            else:
                idle_since.pop(gpu_id, None)

        print(
            "Waiting for an idle GPU among "
            f"{gpu_ids} (current: {', '.join(summaries)})",
            file=sys.stderr,
            flush=True,
        )
        if deadline is not None and now >= deadline:
            raise SystemExit(1)
        time.sleep(max(args.poll_seconds, 1))


if __name__ == "__main__":
    main()
