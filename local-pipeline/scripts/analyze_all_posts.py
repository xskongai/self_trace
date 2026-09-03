#!/usr/bin/env python3
"""SelfTrace batch analyzer v2.0 with provider switching.

Examples:
    # Use config.json (default is OpenAI / gpt-5.6-luna)
    python scripts/analyze_all_posts.py --limit 3

    # Force local Ollama without editing config.json
    python scripts/analyze_all_posts.py --provider local --limit 3

Outputs are separated by provider:
    data/analysis/openai/<post_id>.json
    data/analysis/local/<post_id>.json
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import traceback
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from analyze_post import (
    ANALYSIS_VERSION,
    PROJECT_ROOT,
    SCHEMA_VERSION,
    TqdmLoggingHandler,
    analyze_one_post,
    load_config,
    load_json,
    resolve_project_path,
    resolve_provider_settings,
)

LOGGER = logging.getLogger("selftrace.analyze_all")


def setup_batch_logging(log_file: Path, console_level: str) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.setLevel(logging.DEBUG)
    LOGGER.handlers.clear()
    LOGGER.propagate = False

    console = TqdmLoggingHandler()
    console.setLevel(getattr(logging, console_level.upper(), logging.INFO))
    console.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(funcName)s:%(lineno)d | %(message)s"
    ))
    LOGGER.addHandler(console)
    LOGGER.addHandler(file_handler)


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def output_is_resumable(path: Path, provider: str, model: str) -> bool:
    if not path.exists():
        return False
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return (
            obj.get("schema_version") == SCHEMA_VERSION
            and obj.get("analysis_version") == ANALYSIS_VERSION
            and obj.get("provider") == provider
            and obj.get("model") == model
            and obj.get("analysis_status") in {"OK", "OK_WITH_WARNINGS", "SKIPPED_EMPTY"}
        )
    except Exception:
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SelfTrace batch analyzer v2.0")
    parser.add_argument("--input", default="data/processed/posts_enriched.json", type=Path)
    parser.add_argument("--config", default="config.json", type=Path)
    parser.add_argument("--provider", choices=["openai", "local"], default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    batch_started = time.perf_counter()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    config = load_config(args.config)
    provider, model, provider_settings = resolve_provider_settings(config, args.provider, args.model)

    input_path = resolve_project_path(args.input)
    output_dir = resolve_project_path(args.output_dir) if args.output_dir else (
        PROJECT_ROOT / "data" / "analysis" / provider
    ).resolve()
    log_file = (PROJECT_ROOT / "logs" / f"analyze_all_{provider}_{timestamp}.log").resolve()
    summary_file = (output_dir / f"batch_summary_{timestamp}.json").resolve()
    setup_batch_logging(log_file, args.log_level)

    data = load_json(input_path)
    posts = list(data.get("posts", []))
    if args.limit is not None:
        posts = posts[: max(0, args.limit)]

    LOGGER.info("=" * 84)
    LOGGER.info("SelfTrace batch analyzer v%s", ANALYSIS_VERSION)
    LOGGER.info("Project root : %s", PROJECT_ROOT)
    LOGGER.info("Provider     : %s", provider)
    LOGGER.info("Model        : %s", model)
    LOGGER.info("Input        : %s", input_path)
    LOGGER.info("Output dir   : %s", output_dir)
    LOGGER.info("Selected     : %d posts", len(posts))
    LOGGER.info("Resume       : %s", not args.force)
    LOGGER.info("Log file     : %s", log_file)
    LOGGER.info("=" * 84)

    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    counts = {"success": 0, "warnings": 0, "skipped_existing": 0, "skipped_empty": 0, "failed": 0}
    usage_totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    progress = tqdm(
        total=len(posts),
        desc="Posts",
        unit="post",
        dynamic_ncols=True,
        disable=args.no_progress,
    )

    try:
        for index, post in enumerate(posts, 1):
            post_id = str(post.get("post_id"))
            output_path = (output_dir / f"{post_id}.json").resolve()
            progress.set_description(f"Post {index}/{len(posts)}")
            progress.set_postfix(ok=counts["success"], warn=counts["warnings"], fail=counts["failed"])

            if not args.force and output_is_resumable(output_path, provider, model):
                counts["skipped_existing"] += 1
                results.append({"post_id": post_id, "status": "SKIPPED_EXISTING", "output": str(output_path)})
                LOGGER.info("Skip existing | %d/%d | post=%s", index, len(posts), post_id)
                progress.update(1)
                continue

            LOGGER.info(
                "Post start | %d/%d | post=%s | media=%d | provider=%s | model=%s",
                index, len(posts), post_id, len(post.get("media", [])), provider, model,
            )
            started = time.perf_counter()
            try:
                output = analyze_one_post(
                    post=post,
                    provider=provider,
                    model=model,
                    provider_settings=provider_settings,
                    output_path=output_path,
                    logger=LOGGER,
                    show_progress=False,
                )
                elapsed = time.perf_counter() - started
                status = output.get("analysis_status")
                if status == "SKIPPED_EMPTY":
                    counts["skipped_empty"] += 1
                elif status == "OK_WITH_WARNINGS":
                    counts["warnings"] += 1
                else:
                    counts["success"] += 1

                usage = output.get("usage") or {}
                for key in usage_totals:
                    value = usage.get(key)
                    if isinstance(value, int):
                        usage_totals[key] += value

                results.append({
                    "post_id": post_id,
                    "status": status,
                    "signals": len(output.get("evidence_signals", [])),
                    "elapsed_seconds": round(elapsed, 3),
                    "usage": usage,
                    "output": str(output_path),
                })
                LOGGER.info(
                    "Post done | post=%s | status=%s | signals=%d | %.2fs",
                    post_id, status, len(output.get("evidence_signals", [])), elapsed,
                )
            except Exception as exc:
                elapsed = time.perf_counter() - started
                counts["failed"] += 1
                LOGGER.error("Post failed | post=%s | %.2fs | %s", post_id, elapsed, exc)
                LOGGER.debug(traceback.format_exc())
                results.append({
                    "post_id": post_id,
                    "status": "FAILED",
                    "elapsed_seconds": round(elapsed, 3),
                    "error": str(exc),
                    "output": str(output_path),
                })

            progress.update(1)
            progress.set_postfix(ok=counts["success"], warn=counts["warnings"], fail=counts["failed"])
    finally:
        progress.close()

    elapsed_total = time.perf_counter() - batch_started
    summary = {
        "schema_version": "selftrace-batch-summary-v2.0",
        "analysis_version": ANALYSIS_VERSION,
        "provider": provider,
        "model": model,
        "input": str(input_path),
        "output_dir": str(output_dir),
        "started_at": timestamp,
        "selected_posts": len(posts),
        "counts": counts,
        "usage_totals": usage_totals,
        "elapsed_seconds": round(elapsed_total, 3),
        "results": results,
        "log_file": str(log_file),
    }
    save_json(summary, summary_file)

    LOGGER.info("=" * 84)
    LOGGER.info("Batch completed")
    LOGGER.info("Success          : %d", counts["success"])
    LOGGER.info("Warnings         : %d", counts["warnings"])
    LOGGER.info("Skipped existing : %d", counts["skipped_existing"])
    LOGGER.info("Skipped empty    : %d", counts["skipped_empty"])
    LOGGER.info("Failed           : %d", counts["failed"])
    if provider == "openai":
        LOGGER.info("Input tokens     : %d", usage_totals["input_tokens"])
        LOGGER.info("Output tokens    : %d", usage_totals["output_tokens"])
        LOGGER.info("Total tokens     : %d", usage_totals["total_tokens"])
    LOGGER.info("Elapsed          : %.2fs", elapsed_total)
    LOGGER.info("Summary          : %s", summary_file)
    LOGGER.info("Detailed log     : %s", log_file)
    LOGGER.info("=" * 84)


if __name__ == "__main__":
    main()
