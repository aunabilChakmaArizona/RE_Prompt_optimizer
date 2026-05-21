import argparse
import shlex
from collections import Counter
from pathlib import Path


MODEL_BY_KEY = {
    "qwen": "Qwen/Qwen3-4B",
    "gemma": "google/gemma-3-4b-it",
}

DATASET_BY_CORE = {
    "tacred": "fs_tacred_test_episodes_1shots.pkl",
    "fewrel": "fs_fewrel_test_episodes_1shots.pkl",
}


def parse_prompt_blocks(path):
    blocks = []
    code = None
    status = ""
    lines = []

    for raw_line in Path(path).read_text().splitlines():
        if raw_line.startswith("###### CODE:"):
            if code is not None:
                blocks.append((code, status.strip(), "\n".join(lines).strip()))
            code = raw_line.split(":", 1)[1].strip()
            status = ""
            lines = []
            continue

        if code is None:
            continue

        if raw_line.startswith("###### SOURCE:"):
            continue
        if raw_line.startswith("###### STATUS:"):
            status = raw_line.split(":", 1)[1].strip()
            continue
        lines.append(raw_line)

    if code is not None:
        blocks.append((code, status.strip(), "\n".join(lines).strip()))

    return blocks


def infer_config(code):
    parts = code.split("_")
    if len(parts) < 2:
        raise ValueError(f"Cannot infer dataset/model from CODE: {code}")

    dataset_core = parts[0]
    model_key = parts[1]
    if dataset_core not in DATASET_BY_CORE:
        raise ValueError(f"Unknown dataset prefix in CODE {code!r}: {dataset_core!r}")
    if model_key not in MODEL_BY_KEY:
        raise ValueError(f"Unknown model prefix in CODE {code!r}: {model_key!r}")

    return {
        "dataset_core": dataset_core,
        "dataset": DATASET_BY_CORE[dataset_core],
        "model": MODEL_BY_KEY[model_key],
    }


def shell_join(parts):
    return " ".join(shlex.quote(str(part)) for part in parts)


def make_command(args, code, prompt, config, ordinal, redirect_logs=True):
    log_name = f"{ordinal:03d}_{code}.log".replace("/", "-")
    command = [
        "python",
        "-u",
        args.runner,
        "--model",
        config["model"],
        "--dataset",
        config["dataset"],
        "--dataset_core",
        config["dataset_core"],
        "--ways",
        args.ways,
        "--shots",
        args.shots,
        "--query",
        args.query,
        "--cuda",
        "cuda:0",
        "--ep_start",
        args.ep_start,
        "--ep_end",
        args.ep_end,
        "--batch_size",
        args.batch_size,
        "--data_root",
        args.data_root,
        "--output_dir",
        args.output_dir,
        "--code",
        code,
        "--prompt",
        prompt,
    ]
    if args.background:
        command = ["nohup", *command]
    suffix = " &" if args.background else ""
    redirect = f" > {shlex.quote(str(Path(args.log_dir) / log_name))} 2>&1" if redirect_logs else ""
    return f"{shell_join(command)}{redirect}{suffix}"


def main():
    parser = argparse.ArgumentParser(description="Generate dynamic prompt runner commands.")
    parser.add_argument("--prompts_file", default="all_the_prompts.txt")
    parser.add_argument("--output", default="run_dynamic_prompts.sh")
    parser.add_argument("--plain_output", default=None, help="Optional second script with plain python commands")
    parser.add_argument("--nohup_output", default=None, help="Optional second script with nohup background commands")
    parser.add_argument("--runner", default="codes/final_model_run_icl_dynamic_prompt.py")
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--output_dir", default="outputs/opt_prompt")
    parser.add_argument("--log_dir", default="outputs/logs")
    parser.add_argument("--ways", default="5")
    parser.add_argument("--shots", default="1")
    parser.add_argument("--query", default="0")
    parser.add_argument("--ep_start", default="0")
    parser.add_argument("--ep_end", default="150000")
    parser.add_argument("--batch_size", default="15")
    parser.add_argument("--background", action="store_true", help="Emit nohup background commands instead of sequential commands")
    args = parser.parse_args()

    blocks = parse_prompt_blocks(args.prompts_file)
    status_blocked_count = sum(1 for _, status, _ in blocks if status.lower() in {"running", "done"})
    other_status_count = sum(
        1
        for _, status, _ in blocks
        if status.strip() and status.lower() not in {"running", "done"}
    )
    non_empty_blocks = [
        (code, prompt)
        for code, status, prompt in blocks
        if not status.strip() and prompt
    ]
    empty_count = sum(1 for _, status, prompt in blocks if not status.strip() and not prompt)

    seen_pairs = set()
    command_payloads = []
    skipped_duplicate_count = 0
    code_counts = Counter()

    for code, prompt in non_empty_blocks:
        pair = (code, prompt)
        if pair in seen_pairs:
            skipped_duplicate_count += 1
            continue
        seen_pairs.add(pair)
        code_counts[code] += 1
        run_code = code if code_counts[code] == 1 else f"{code}__dup{code_counts[code]}"
        config = infer_config(code)
        command_payloads.append((run_code, prompt, config))

    def render_command(payload, ordinal, background):
        original_background = args.background
        args.background = background
        try:
            return make_command(
                args,
                payload[0],
                payload[1],
                payload[2],
                ordinal,
                redirect_logs=background,
            )
        finally:
            args.background = original_background

    def write_script(path, background):
        run_mode_comments = (
            [
                "# Commands run with nohup in the background.",
                "# All commands still use cuda:0.",
            ]
            if background
            else [
                "# Plain python commands. Run this script to execute commands sequentially.",
                "# No shell log redirection is added in this script.",
                "# All commands use cuda:0.",
            ]
        )
        commands = [
            render_command(payload, index + 1, background)
            for index, payload in enumerate(command_payloads)
        ]
        path = Path(path)
        path.write_text(
            "\n".join(
                [
                    "#!/usr/bin/env bash",
                    "set -euo pipefail",
                    f"mkdir -p {shlex.quote(args.log_dir)} {shlex.quote(args.output_dir)}",
                    "",
                    f"# Generated from {args.prompts_file}",
                    "# Each CODE block is passed as the instruction split; answer/input splits come from agents.agent_prompts.",
                    *run_mode_comments,
                    f"# Prompt blocks: {len(blocks)}",
                    f"# Empty-status non-empty prompt blocks: {len(non_empty_blocks)}",
                    f"# Empty-status empty prompt blocks skipped: {empty_count}",
                    f"# Status running/done blocks skipped: {status_blocked_count}",
                    f"# Other non-empty status blocks skipped: {other_status_count}",
                    f"# Exact duplicate non-empty blocks skipped: {skipped_duplicate_count}",
                    "",
                    *commands,
                    "",
                ]
            )
        )
        path.chmod(0o755)
        return path, len(commands)

    output = Path(args.output)
    written = [write_script(output, args.background)]
    if args.plain_output:
        written.append(write_script(Path(args.plain_output), False))
    if args.nohup_output:
        written.append(write_script(Path(args.nohup_output), True))

    # Kept as a single user-facing summary even when multiple scripts are written.
    for path, count in written:
        print(f"Wrote {count} commands to {path}")
    print(
        f"Skipped {empty_count} empty blocks, {status_blocked_count} running/done blocks, "
        f"{other_status_count} other non-empty-status blocks, "
        f"and {skipped_duplicate_count} exact duplicate non-empty blocks"
    )


if __name__ == "__main__":
    main()
