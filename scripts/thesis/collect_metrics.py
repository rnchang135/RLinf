#!/usr/bin/env python3

import argparse
import csv
import re
from pathlib import Path


ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
STEP_RE = re.compile(r"Global Step:\s*(\d+)\s*/\s*(\d+)")
STEP_TIME_RE = re.compile(r"Step Time:\s*([0-9.]+)s")
METRIC_RE = re.compile(
    r"([A-Za-z_][A-Za-z0-9_./]*)="
    r"(-?(?:nan|inf|(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?))",
    re.IGNORECASE,
)


def clean_line(line: str) -> str:
    return ANSI_RE.sub("", line).replace("\r", "")


def metric_column(section: str, key: str) -> str:
    key = key.replace("/", "_").replace(".", "_")

    if key.startswith("actor_") or key.startswith("critic_"):
        return key

    prefix = {
        "environment": "env",
        "evaluation": "eval",
        "rollout": "rollout",
        "actor": "actor",
        "critic": "critic",
        "time": "time",
    }.get(section, "other")

    return f"{prefix}_{key}"


def parse_log(log_path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    section = ""

    with log_path.open("r", encoding="utf-8", errors="replace") as file:
        for raw_line in file:
            line = clean_line(raw_line)

            step_match = STEP_RE.search(line)
            if step_match:
                if current is not None:
                    rows.append(current)

                current = {
                    "global_step": step_match.group(1),
                    "total_steps": step_match.group(2),
                }
                section = ""
                continue

            if current is None:
                continue

            if "Training/Actor" in line:
                section = "actor"
            elif "Training/Critic" in line:
                section = "critic"
            elif "Environment" in line:
                section = "environment"
            elif "Evaluation" in line:
                section = "evaluation"
            elif "Rollout" in line:
                section = "rollout"
            elif "Time" in line:
                section = "time"

            step_time_match = STEP_TIME_RE.search(line)
            if step_time_match:
                current["step_time_seconds"] = step_time_match.group(1)

            for key, value in METRIC_RE.findall(line):
                current[metric_column(section, key)] = value

    if current is not None:
        rows.append(current)

    return rows


def write_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    preferred_columns = [
        "global_step",
        "total_steps",
        "step_time_seconds",
        "env_num_trajectories",
        "env_success_once",
        "env_return",
        "env_reward",
        "eval_num_trajectories",
        "eval_success_once",
        "eval_success_at_end",
        "eval_return",
        "eval_reward",
        "actor_approx_kl",
        "actor_clip_fraction",
        "actor_grad_norm",
        "actor_policy_loss",
        "actor_total_loss",
        "actor_lr",
        "critic_value_loss",
        "critic_explained_variance",
        "critic_lr",
    ]

    discovered = {key for row in rows for key in row}
    columns = [column for column in preferred_columns if column in discovered]
    columns.extend(sorted(discovered - set(columns)))

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract RLinf training metrics into a CSV file."
    )
    parser.add_argument("log_file", help="Path to launcher_console.log")
    parser.add_argument(
        "--output",
        help="Output CSV path; defaults to summary.csv beside the log",
    )
    args = parser.parse_args()

    log_path = Path(args.log_file).expanduser().resolve()
    if not log_path.is_file():
        raise SystemExit(f"[ERROR] 日志文件不存在：{log_path}")

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else log_path.with_name("summary.csv")
    )

    rows = parse_log(log_path)
    if not rows:
        raise SystemExit("[ERROR] 没有在日志中找到 Global Step 指标。")

    write_csv(rows, output_path)

    print(f"[OK] 提取到 {len(rows)} 个 Global Step")
    print(f"[OK] CSV 文件：{output_path}")


if __name__ == "__main__":
    main()
