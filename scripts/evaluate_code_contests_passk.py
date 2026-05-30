from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from _bootstrap import PROJECT_ROOT  # noqa: F401

from config import load_yaml
from jsonl_io import iter_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="configs/experiments/code_contests/code_contests_full_v1_manifest.json",
    )
    parser.add_argument(
        "--output-dir",
        default="results/analysis/code_contests/passk_v1",
    )
    parser.add_argument("--run-names", default=None)
    parser.add_argument("--k-values", default="1,2,4,8,16")
    parser.add_argument("--test-suites", default="public,private")
    parser.add_argument("--max-workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    parser.add_argument("--timeout-multiplier", type=float, default=2.0)
    parser.add_argument("--min-timeout-seconds", type=float, default=2.0)
    parser.add_argument("--max-timeout-seconds", type=float, default=12.0)
    parser.add_argument("--memory-limit-mb", type=int, default=2048)
    parser.add_argument("--limit-prompts", type=int, default=None)
    return parser.parse_args()


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    return slug or "value"


def _load_manifest(manifest_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        manifest = dict(payload)
        experiments = [dict(experiment) for experiment in manifest.get("experiments", [])]
    elif isinstance(payload, list):
        manifest = {"experiments": payload}
        experiments = [dict(experiment) for experiment in payload]
    else:
        raise TypeError(f"Unsupported manifest payload type: {type(payload)!r}")

    for experiment in experiments:
        if "model_key" not in experiment or not experiment.get("model_key"):
            config = load_yaml(experiment["config_path"])
            model_cfg = config.get("model", {})
            experiment["model_key"] = model_cfg.get("key") or model_cfg.get("hf_id") or "unknown_model"
        experiment.setdefault("dataset_label", "code_contests")
        experiment.setdefault("stage_name", "code_contests_full")
    manifest["experiments"] = experiments
    return manifest, experiments


def _selected_run_names(args: argparse.Namespace, manifest: dict[str, Any]) -> list[str]:
    if args.run_names:
        return [item.strip() for item in args.run_names.split(",") if item.strip()]
    return [experiment["run_name"] for experiment in manifest["experiments"]]


def _parse_k_values(raw: str) -> list[int]:
    values = sorted({int(chunk.strip()) for chunk in raw.split(",") if chunk.strip()})
    if not values:
        raise ValueError("At least one k value is required.")
    return values


def _parse_test_suites(raw: str) -> list[str]:
    suites = [chunk.strip().lower() for chunk in raw.split(",") if chunk.strip()]
    if not suites:
        raise ValueError("At least one test suite is required.")
    allowed = {"public", "private", "generated"}
    invalid = [suite for suite in suites if suite not in allowed]
    if invalid:
        raise ValueError(f"Unsupported test suites: {invalid}")
    return suites


def _load_prompt_metadata(config_path: str | Path) -> dict[int, dict[str, Any]]:
    config = load_yaml(config_path)
    prepared_path = Path(str(config["data"]["prepared_path"]).replace("${USER}", os.environ["USER"]))
    prompt_meta: dict[int, dict[str, Any]] = {}
    for pool in iter_jsonl(prepared_path):
        if pool.get("split") != "test":
            continue
        prompt_meta[int(pool["prompt_id"])] = {
            "source_index": int(pool["source_index"]),
            "prompt": pool["prompt"],
            "prepared_path": str(prepared_path),
        }
    return prompt_meta


def _load_code_contests_dataset(hf_split: str) -> Any:
    from datasets import Dataset, concatenate_datasets

    cache_root = Path(
        os.environ.get(
            "HF_DATASETS_CACHE",
            f"${SCRATCH_ROOT_BASE}/{os.environ['USER']}/cache/huggingface/datasets",
        )
    )
    dataset_dirs = sorted(cache_root.glob("deepmind___code_contests/default/*/*"))
    if not dataset_dirs:
        raise FileNotFoundError(f"Could not find cached deepmind/code_contests dataset under {cache_root}")

    split_name = str(hf_split).lower()
    if split_name == "validation":
        split_name = "valid"
    if split_name == "train":
        pattern = "code_contests-train-*.arrow"
    elif split_name == "valid":
        pattern = "code_contests-valid.arrow"
    elif split_name == "test":
        pattern = "code_contests-test.arrow"
    else:
        raise ValueError(f"Unsupported code_contests split: {hf_split}")

    shard_paths: list[Path] = []
    for dataset_dir in dataset_dirs:
        shard_paths.extend(sorted(dataset_dir.glob(pattern)))
    if not shard_paths:
        raise FileNotFoundError(f"Could not find cached Arrow shards for split '{hf_split}'")

    datasets = [Dataset.from_file(str(path)) for path in shard_paths]
    if len(datasets) == 1:
        return datasets[0]
    return concatenate_datasets(datasets)


def _collect_problem_specs(
    *,
    config_path: str | Path,
    prompt_meta: dict[int, dict[str, Any]],
    test_suites: list[str],
    limit_prompts: int | None,
) -> dict[int, dict[str, Any]]:
    config = load_yaml(config_path)
    dataset = _load_code_contests_dataset(str(config["data"]["hf_split"]))
    selected_prompt_ids = sorted(prompt_meta)
    if limit_prompts is not None:
        selected_prompt_ids = selected_prompt_ids[:limit_prompts]

    specs: dict[int, dict[str, Any]] = {}
    for prompt_id in selected_prompt_ids:
        meta = prompt_meta[prompt_id]
        row = dataset[int(meta["source_index"])]
        tests: list[dict[str, str]] = []
        for suite_name in test_suites:
            suite_key = f"{suite_name}_tests"
            suite = row.get(suite_key) or {}
            inputs = suite.get("input") or []
            outputs = suite.get("output") or []
            tests.extend(
                {
                    "suite": suite_name,
                    "input": str(test_input),
                    "output": str(test_output),
                }
                for test_input, test_output in zip(inputs, outputs)
            )
        specs[prompt_id] = {
            "prompt_id": prompt_id,
            "prompt": meta["prompt"],
            "source_index": meta["source_index"],
            "tests": tests,
            "time_limit_seconds": float(row.get("time_limit", {}).get("seconds") or 0.0),
            "memory_limit_bytes": int(row.get("memory_limit_bytes") or 0),
            "input_file": (row.get("input_file") or "").strip(),
            "output_file": (row.get("output_file") or "").strip(),
        }
    return specs


def _group_generations(path: Path, limit_prompts: int | None) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for record in iter_jsonl(path):
        prompt_id = int(record["prompt_id"])
        grouped.setdefault(prompt_id, []).append(record)
    for prompt_id in grouped:
        grouped[prompt_id] = sorted(grouped[prompt_id], key=lambda item: int(item.get("generation_index", 0)))
    if limit_prompts is None:
        return grouped
    limited_ids = sorted(grouped)[:limit_prompts]
    return {prompt_id: grouped[prompt_id] for prompt_id in limited_ids}


def _normalize_text(value: str) -> str:
    lines = [line.rstrip() for line in value.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    return "\n".join(lines).strip()


def _tokens_match(expected_tokens: list[str], actual_tokens: list[str]) -> bool:
    if len(expected_tokens) != len(actual_tokens):
        return False
    for expected, actual in zip(expected_tokens, actual_tokens):
        if expected == actual:
            continue
        try:
            if not math.isclose(float(expected), float(actual), rel_tol=1e-6, abs_tol=1e-6):
                return False
        except ValueError:
            return False
    return True


def _outputs_match(expected: str, actual: str) -> bool:
    expected_normalized = _normalize_text(expected)
    actual_normalized = _normalize_text(actual)
    if expected_normalized == actual_normalized:
        return True
    expected_tokens = expected_normalized.split()
    actual_tokens = actual_normalized.split()
    return _tokens_match(expected_tokens, actual_tokens)


def _extract_code(response: str) -> str:
    text = response.strip()
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()

    fenced = re.findall(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return max((block.strip() for block in fenced), key=len)

    lines = text.splitlines()
    for index, line in enumerate(lines):
        if re.match(r"^\s*(from\s+\w+|import\s+\w+|def\s+\w+|class\s+\w+|if __name__ == ['\"]__main__['\"]:)", line):
            return "\n".join(lines[index:]).strip()
    return text


def _preexec_with_limits(memory_limit_mb: int, cpu_timeout_seconds: float) -> Any:
    def _fn() -> None:
        import resource

        memory_bytes = int(memory_limit_mb) * 1024 * 1024
        cpu_limit = max(1, int(math.ceil(cpu_timeout_seconds)) + 1)
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))
        resource.setrlimit(resource.RLIMIT_FSIZE, (50 * 1024 * 1024, 50 * 1024 * 1024))
        resource.setrlimit(resource.RLIMIT_NPROC, (32, 32))

    return _fn


def _run_single_test(
    *,
    code: str,
    test_input: str,
    expected_output: str,
    problem_spec: dict[str, Any],
    timeout_seconds: float,
    memory_limit_mb: int,
) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory(prefix="mrt_code_eval_", dir="/tmp") as tmp_dir:
        temp_path = Path(tmp_dir)
        code_path = temp_path / "candidate.py"
        code_path.write_text(code, encoding="utf-8")

        stdin_bytes = None
        input_file = str(problem_spec.get("input_file") or "")
        output_file = str(problem_spec.get("output_file") or "")
        if input_file:
            (temp_path / input_file).write_text(test_input, encoding="utf-8")
        else:
            stdin_bytes = test_input.encode("utf-8")

        command = [sys.executable, "-I", "-B", str(code_path)]
        try:
            result = subprocess.run(
                command,
                input=stdin_bytes,
                capture_output=True,
                cwd=temp_path,
                timeout=timeout_seconds,
                preexec_fn=_preexec_with_limits(memory_limit_mb, timeout_seconds),
            )
        except subprocess.TimeoutExpired:
            return False, "timeout"
        except Exception:
            return False, "judge_error"

        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="ignore")
            if "SyntaxError" in stderr or "IndentationError" in stderr:
                return False, "compile_error"
            return False, "runtime_error"

        if output_file and (temp_path / output_file).exists():
            actual_output = (temp_path / output_file).read_text(encoding="utf-8", errors="ignore")
        else:
            actual_output = result.stdout.decode("utf-8", errors="ignore")
        if _outputs_match(expected_output, actual_output):
            return True, "passed"
        return False, "wrong_answer"


def _effective_timeout(problem_spec: dict[str, Any], args: argparse.Namespace) -> float:
    base = float(problem_spec.get("time_limit_seconds") or 0.0)
    if base <= 0:
        base = float(args.min_timeout_seconds)
    timeout_seconds = max(args.min_timeout_seconds, min(args.max_timeout_seconds, base * args.timeout_multiplier))
    return float(timeout_seconds)


def _evaluate_candidate(
    *,
    response: str,
    problem_spec: dict[str, Any],
    args_dict: dict[str, Any],
) -> dict[str, Any]:
    code = _extract_code(response)
    if not code.strip():
        return {"passed": False, "status": "empty"}

    timeout_seconds = _effective_timeout(problem_spec, argparse.Namespace(**args_dict))
    for test in problem_spec["tests"]:
        passed, status = _run_single_test(
            code=code,
            test_input=test["input"],
            expected_output=test["output"],
            problem_spec=problem_spec,
            timeout_seconds=timeout_seconds,
            memory_limit_mb=int(args_dict["memory_limit_mb"]),
        )
        if not passed:
            return {"passed": False, "status": status}
    return {"passed": True, "status": "passed"}


def _evaluate_prompt(task: dict[str, Any]) -> dict[str, Any]:
    args_dict = task["args"]
    problem_spec = task["problem_spec"]
    prompt_records = task["records"]
    statuses: dict[str, int] = {}
    pass_count = 0
    for record in prompt_records:
        outcome = _evaluate_candidate(
            response=record["response"],
            problem_spec=problem_spec,
            args_dict=args_dict,
        )
        pass_count += int(outcome["passed"])
        statuses[outcome["status"]] = statuses.get(outcome["status"], 0) + 1
    return {
        "prompt_id": int(problem_spec["prompt_id"]),
        "source_index": int(problem_spec["source_index"]),
        "num_samples": len(prompt_records),
        "num_passed": pass_count,
        "statuses": statuses,
    }


def _pass_at_k(num_samples: int, num_correct: int, k: int) -> float:
    if num_correct <= 0:
        return 0.0
    if num_samples - num_correct < k:
        return 1.0
    product = 1.0
    for index in range(k):
        product *= (num_samples - num_correct - index) / (num_samples - index)
    return 1.0 - product


def main() -> None:
    args = parse_args()
    manifest_path = PROJECT_ROOT / args.manifest
    manifest, experiments = _load_manifest(manifest_path)
    run_names = set(_selected_run_names(args, manifest))
    selected_experiments = [experiment for experiment in experiments if experiment["run_name"] in run_names]
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    per_run_dir = output_dir / "per_run"
    per_run_dir.mkdir(parents=True, exist_ok=True)

    k_values = _parse_k_values(args.k_values)
    test_suites = _parse_test_suites(args.test_suites)
    args_dict = {
        "timeout_multiplier": args.timeout_multiplier,
        "min_timeout_seconds": args.min_timeout_seconds,
        "max_timeout_seconds": args.max_timeout_seconds,
        "memory_limit_mb": args.memory_limit_mb,
    }

    run_rows: list[dict[str, Any]] = []
    overall_started_at = time.perf_counter()
    for index, experiment in enumerate(selected_experiments, start=1):
        run_name = str(experiment["run_name"])
        run_dir = PROJECT_ROOT / "results" / "runs" / run_name
        generation_path = run_dir / "multi_generations.jsonl"
        if not generation_path.exists():
            print(f"[{index}/{len(selected_experiments)}] Missing {generation_path}; skipping {run_name}.")
            continue

        prompt_meta = _load_prompt_metadata(experiment["config_path"])
        prompt_specs = _collect_problem_specs(
            config_path=experiment["config_path"],
            prompt_meta=prompt_meta,
            test_suites=test_suites,
            limit_prompts=args.limit_prompts,
        )
        generations_by_prompt = _group_generations(generation_path, args.limit_prompts)
        tasks = [
            {
                "problem_spec": prompt_specs[prompt_id],
                "records": generations_by_prompt[prompt_id],
                "args": args_dict,
            }
            for prompt_id in sorted(generations_by_prompt)
            if prompt_id in prompt_specs and prompt_specs[prompt_id]["tests"]
        ]
        run_started_at = time.perf_counter()
        if not tasks:
            print(f"[{index}/{len(selected_experiments)}] No tasks for {run_name}; skipping.")
            continue

        with ProcessPoolExecutor(max_workers=int(args.max_workers), mp_context=mp.get_context("spawn")) as executor:
            prompt_rows = list(executor.map(_evaluate_prompt, tasks))

        status_totals: dict[str, int] = {}
        run_row = {
            "run_name": run_name,
            "model_key": experiment.get("model_key"),
            "dataset_label": experiment.get("dataset_label"),
            "stage_name": experiment.get("stage_name"),
            "strategy": experiment.get("strategy"),
            "k": int(experiment.get("k")),
            "test_suites": ",".join(test_suites),
            "num_prompts": len(prompt_rows),
            "num_samples_total": int(sum(row["num_samples"] for row in prompt_rows)),
            "num_correct_total": int(sum(row["num_passed"] for row in prompt_rows)),
        }
        for prompt_row in prompt_rows:
            for status, count in prompt_row["statuses"].items():
                status_totals[status] = status_totals.get(status, 0) + int(count)

        for k_value in k_values:
            estimates = [
                _pass_at_k(int(prompt_row["num_samples"]), int(prompt_row["num_passed"]), k_value)
                for prompt_row in prompt_rows
                if int(prompt_row["num_samples"]) >= k_value
            ]
            run_row[f"pass_at_{k_value}"] = float(sum(estimates) / len(estimates)) if estimates else 0.0
        run_row["mean_passed_per_prompt"] = float(
            sum(prompt_row["num_passed"] for prompt_row in prompt_rows) / max(len(prompt_rows), 1)
        )
        run_row["sample_pass_rate_mean"] = float(
            run_row["num_correct_total"] / max(run_row["num_samples_total"], 1)
        )
        run_row["elapsed_seconds"] = time.perf_counter() - run_started_at
        run_row["status_counts_json"] = json.dumps(status_totals, sort_keys=True)

        per_run_payload = {
            "run_name": run_name,
            "config_path": experiment["config_path"],
            "test_suites": test_suites,
            "summary": run_row,
            "prompt_rows": prompt_rows,
            "status_totals": status_totals,
        }
        (per_run_dir / f"{run_name}.json").write_text(json.dumps(per_run_payload, indent=2), encoding="utf-8")
        run_rows.append(run_row)
        print(
            f"[{index}/{len(selected_experiments)}] Evaluated {run_name}: "
            f"{run_row['num_prompts']} prompts, pass@16={run_row.get('pass_at_16', 0.0):.4f}"
        )

    import pandas as pd

    summary_frame = pd.DataFrame(run_rows).sort_values(["strategy", "k", "run_name"]).reset_index(drop=True)
    summary_csv = output_dir / "passk_summary.csv"
    summary_frame.to_csv(summary_csv, index=False)
    metadata = {
        "manifest": args.manifest,
        "test_suites": test_suites,
        "k_values": k_values,
        "max_workers": int(args.max_workers),
        "timeout_multiplier": float(args.timeout_multiplier),
        "min_timeout_seconds": float(args.min_timeout_seconds),
        "max_timeout_seconds": float(args.max_timeout_seconds),
        "memory_limit_mb": int(args.memory_limit_mb),
        "limit_prompts": args.limit_prompts,
        "num_runs": len(run_rows),
        "elapsed_seconds": time.perf_counter() - overall_started_at,
        "summary_csv": str(summary_csv),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(summary_csv)


if __name__ == "__main__":
    main()
