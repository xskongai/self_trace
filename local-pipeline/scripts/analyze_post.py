#!/usr/bin/env python3
"""SelfTrace single-post analyzer v2.0.

v2.0 keeps two interchangeable inference backends:

    provider=openai -> gpt-5.6-luna (default hosted backend)
    provider=local  -> qwen3-vl:8b through Ollama

Both providers use the SAME one-shot multimodal schema:

    post text + all local images
        -> one model call
        -> media observations + post understanding + candidate signals
        -> deterministic Python validator
        -> final auditable JSON

All relative paths are resolved from the local-pipeline project root. The
Terminal working directory therefore does not change where files are read/written.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import mimetypes
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from tqdm import tqdm

from signal_validator import normalize_post_intents, validate_signal_dicts


# -----------------------------------------------------------------------------
# Version / paths
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_VERSION = "2.0"
SCHEMA_VERSION = "selftrace-analysis-v2.0"
LOGGER = logging.getLogger("selftrace.analyze_post")

# Load secrets only from the project-local file or the process environment.
# The key is never printed by this program.
load_dotenv(PROJECT_ROOT / ".env.local", override=False)


# -----------------------------------------------------------------------------
# Structured output schema shared by BOTH providers
# -----------------------------------------------------------------------------

MEDIA_TYPES = Literal[
    "PHOTO", "SCREENSHOT", "PAPER_SCREENSHOT", "CHART", "TABLE", "DIAGRAM",
    "CODE", "FORMULA", "NOTE", "MEME", "OTHER"
]
POST_TYPES = Literal[
    "LEARNING_NOTE", "RESEARCH_NOTE", "PERSONAL_REFLECTION", "PROJECT_UPDATE",
    "QUESTION", "OPINION", "OBSERVATION", "PHOTO_SHARE", "RESOURCE_SHARE", "OTHER"
]
INTENTS = Literal[
    "RECORD_IDEA", "SHARE_RESULT", "ASK_QUESTION", "REFLECT", "LEARN", "EXPLAIN",
    "COMPARE", "EXPRESS_FEELING", "DOCUMENT_PROGRESS", "OTHER"
]
DIMENSIONS = Literal[
    "INTEREST", "KNOWLEDGE", "ACTIVITY", "THINKING_PATTERN", "EXPRESSION_STYLE", "GOAL_ORIENTATION"
]
CONFIDENCE = Literal["HIGH", "MEDIUM"]


class ChartObservation(BaseModel):
    title: str | None = None
    chart_type: str | None = None
    x_axis: str | None = None
    y_axis: str | None = None
    entities: list[str] = Field(default_factory=list)
    main_pattern: str | None = None
    notable_values: list[str] = Field(default_factory=list)


class MediaObservation(BaseModel):
    # Must be copied from the media mapping in the prompt.
    media_index: int
    media_type: MEDIA_TYPES
    description: str
    visible_text: list[str] = Field(default_factory=list)
    concepts: list[str] = Field(default_factory=list)
    visual_elements: list[str] = Field(default_factory=list)
    chart: ChartObservation | None = None
    uncertainty: list[str] = Field(default_factory=list)


class PostUnderstanding(BaseModel):
    content_summary: str
    primary_topics: list[str] = Field(default_factory=list)
    secondary_topics: list[str] = Field(default_factory=list)
    post_types: list[POST_TYPES] = Field(default_factory=list)
    intents: list[INTENTS] = Field(default_factory=list)
    source_relationship: Literal[
        "AUTHOR_ORIGINAL", "COMMENTARY_ON_EXTERNAL_SOURCE",
        "SHARED_SOURCE_WITHOUT_CLEAR_COMMENTARY", "UNCLEAR"
    ]
    explicit_statements: list[str] = Field(default_factory=list)
    uncertainty: list[str] = Field(default_factory=list)


class EvidenceSignal(BaseModel):
    dimension: DIMENSIONS
    claim: str
    evidence: list[str]
    confidence: CONFIDENCE
    temporal_scope: Literal["THIS_POST_ONLY"] = "THIS_POST_ONLY"


class UnifiedPostAnalysis(BaseModel):
    media_observations: list[MediaObservation] = Field(default_factory=list)
    post_understanding: PostUnderstanding
    evidence_signals: list[EvidenceSignal] = Field(default_factory=list)
    rejected_inferences: list[str] = Field(default_factory=list)


SYSTEM_PROMPT = """You are SelfTrace's evidence extraction engine.
Analyze ONE social-media post using the author's raw text and all attached images together.

You must return four things in one structured response:
1. objective media observations for every supplied image;
2. post-level understanding;
3. 0-4 narrow evidence signals;
4. rejected/unsafe or unsupported inferences when relevant.

MEDIA RULES
- Each supplied image has a media_index. Return exactly one media_observation for each supplied image and preserve its media_index.
- Describe only what is supported by pixels.
- For text-heavy screenshots/notes, visible_text must contain representative legible text and concepts must contain the main concepts.
- NOTE = explanatory slide / note / bullet-note screenshot.
- PAPER_SCREENSHOT only when there is clear evidence of an academic-paper page.
- TABLE only when there are genuine rows/columns.
- Never invent unreadable text or numbers; use uncertainty instead.

POST RULES
- primary_topics should contain 1-5 concrete topics when interpretable.
- post_types and intents should be non-empty when interpretable.
- explicit_statements should preserve explicit author preferences/goals/claims.
- Distinguish author text from material merely displayed in images.

SIGNAL RULES
- INTEREST: explicit attraction/fascination/preference/curiosity toward a topic.
- KNOWLEDGE: demonstrated technical understanding, not mere exposure.
- ACTIVITY: studying, reading, implementing, experimenting, documenting work.
- THINKING_PATTERN: reasoning operation shown in THIS post (comparison, abstraction, analogy, mechanism-seeking, pattern-linking). Do not phrase it as a permanent trait.
- EXPRESSION_STYLE: how the author communicates, not what they like.
- GOAL_ORIENTATION: explicit future intention, plan, learning/research goal.
- HIGH = directly explicit in author text; MEDIUM = reasonable inference from this post.
- Do not create a signal whose only content is 'the author attached an image showing X'.
- Do not create near-duplicate signals.
- Never infer sensitive attributes, diagnoses, romantic feelings, or stable personality from a post.
"""


# -----------------------------------------------------------------------------
# Logging helpers
# -----------------------------------------------------------------------------

class TqdmLoggingHandler(logging.Handler):
    """Write console logs without breaking tqdm bars."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.write(self.format(record), file=sys.stderr)
        except Exception:
            self.handleError(record)


def setup_logging(log_file: Path, console_level: str = "INFO", logger: logging.Logger = LOGGER) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    console = TqdmLoggingHandler()
    console.setLevel(getattr(logging, console_level.upper(), logging.INFO))
    console.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(funcName)s:%(lineno)d | %(message)s"
    ))

    logger.addHandler(console)
    logger.addHandler(file_handler)


# -----------------------------------------------------------------------------
# Files / config
# -----------------------------------------------------------------------------

def resolve_project_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    return p.resolve() if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_config(path: str | Path = "config.json") -> dict:
    config_path = resolve_project_path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    return load_json(config_path)


def find_post(data: dict, post_id: str | None) -> dict:
    posts = data.get("posts", [])
    if post_id:
        for post in posts:
            if str(post.get("post_id")) == str(post_id):
                return post
        raise ValueError(f"Post not found: {post_id}")
    for post in posts:
        if post.get("media") or (post.get("text") or "").strip():
            return post
    raise ValueError("No analyzable post found")


def resolve_provider_settings(config: dict, provider_override: str | None, model_override: str | None) -> tuple[str, str, dict]:
    provider = (provider_override or config.get("provider") or "openai").lower()
    if provider not in {"openai", "local"}:
        raise ValueError("provider must be 'openai' or 'local'")

    section = dict(config.get(provider) or {})
    default_model = "gpt-5.6-luna" if provider == "openai" else "qwen3-vl:8b"
    model = model_override or section.get("model") or default_model
    return provider, model, section


def available_media(post: dict, logger: logging.Logger) -> tuple[list[Path], list[dict], list[dict]]:
    """Return (existing image paths, model mapping, unavailable observations)."""
    paths: list[Path] = []
    mapping: list[dict] = []
    unavailable: list[dict] = []

    for position, media in enumerate(post.get("media", []), 1):
        media_index = int(media.get("index", position - 1))
        local_path = media.get("local_path")
        if not local_path or media.get("download_status") == "failed":
            reason = media.get("download_error") or "missing local_path"
            unavailable.append({"media_index": media_index, "status": "unavailable", "reason": reason})
            logger.warning("Media unavailable | index=%s | %s", media_index, reason)
            continue

        path = resolve_project_path(local_path)
        if not path.exists():
            reason = f"file not found: {path}"
            unavailable.append({"media_index": media_index, "status": "unavailable", "reason": reason})
            logger.warning("Media unavailable | index=%s | %s", media_index, reason)
            continue

        paths.append(path)
        mapping.append({
            "image_order": len(paths),
            "media_index": media_index,
            "file_name": path.name,
        })

    return paths, mapping, unavailable


# -----------------------------------------------------------------------------
# Provider calls
# -----------------------------------------------------------------------------

def build_user_prompt(post: dict, media_mapping: list[dict]) -> str:
    payload = {
        "post_id": str(post.get("post_id")),
        "created_at": post.get("created_at"),
        "raw_text": post.get("text") or "",
        "image_mapping": media_mapping,
    }
    return (
        "Analyze this SelfTrace post. Images are supplied after this text in the exact order shown in image_mapping. "
        "Copy each media_index into the corresponding media_observation.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


def image_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def extract_parsed_response(response) -> UnifiedPostAnalysis:
    """Extract the Pydantic object produced by OpenAI responses.parse()."""
    for output in response.output:
        if getattr(output, "type", None) != "message":
            continue
        for item in getattr(output, "content", []):
            if getattr(item, "type", None) == "output_text" and getattr(item, "parsed", None) is not None:
                return item.parsed
    raise RuntimeError("OpenAI response completed but no parsed structured output was found")


def call_openai(
    model: str,
    post: dict,
    images: list[Path],
    media_mapping: list[dict],
    image_detail: str,
    retries: int,
    logger: logging.Logger,
) -> tuple[UnifiedPostAnalysis, dict]:
    """Hosted backend: one Responses API call for the whole post."""
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not available. Keep the key local in local-pipeline/.env.local "
            "or your shell environment; do not paste it into chat."
        )

    from openai import OpenAI

    client = OpenAI()
    content: list[dict] = [{"type": "input_text", "text": build_user_prompt(post, media_mapping)}]
    for path in images:
        content.append({
            "type": "input_image",
            "image_url": image_data_url(path),
            "detail": image_detail,
        })

    last_error: Exception | None = None
    for attempt in range(1, retries + 2):
        started = time.perf_counter()
        logger.info(
            "OpenAI start | attempt=%d/%d | model=%s | images=%d | detail=%s",
            attempt, retries + 1, model, len(images), image_detail,
        )
        try:
            response = client.responses.parse(
                model=model,
                instructions=SYSTEM_PROMPT,
                input=[{"role": "user", "content": content}],
                text_format=UnifiedPostAnalysis,
                reasoning={"effort": "none"},
            )
            parsed = extract_parsed_response(response)
            elapsed = time.perf_counter() - started

            usage_obj = getattr(response, "usage", None)
            usage = {
                "input_tokens": getattr(usage_obj, "input_tokens", None),
                "output_tokens": getattr(usage_obj, "output_tokens", None),
                "total_tokens": getattr(usage_obj, "total_tokens", None),
            }
            logger.info(
                "OpenAI done  | %.2fs | input_tokens=%s | output_tokens=%s",
                elapsed, usage["input_tokens"], usage["output_tokens"],
            )
            return parsed, usage
        except Exception as exc:
            last_error = exc
            logger.warning("OpenAI failed | attempt=%d | %s", attempt, exc)
            logger.debug(traceback.format_exc())
            if attempt <= retries:
                time.sleep(min(2.0 * attempt, 5.0))

    raise RuntimeError(f"OpenAI call failed after {retries + 1} attempts: {last_error}")


def call_local(
    model: str,
    post: dict,
    images: list[Path],
    media_mapping: list[dict],
    retries: int,
    logger: logging.Logger,
) -> tuple[UnifiedPostAnalysis, dict]:
    """Local backend: one Ollama multimodal call for the whole post."""
    from ollama import chat

    message: dict = {
        "role": "user",
        "content": build_user_prompt(post, media_mapping),
    }
    if images:
        message["images"] = [str(path) for path in images]

    last_error: Exception | None = None
    for attempt in range(1, retries + 2):
        started = time.perf_counter()
        logger.info(
            "Ollama start | attempt=%d/%d | model=%s | images=%d",
            attempt, retries + 1, model, len(images),
        )
        try:
            response = chat(
                model=model,
                format=UnifiedPostAnalysis.model_json_schema(),
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    message,
                ],
                options={"temperature": 0},
            )
            parsed = UnifiedPostAnalysis.model_validate_json(response.message.content)
            elapsed = time.perf_counter() - started
            logger.info("Ollama done  | %.2fs | chars=%d", elapsed, len(response.message.content or ""))
            return parsed, {}
        except Exception as exc:
            last_error = exc
            logger.warning("Ollama failed | attempt=%d | %s", attempt, exc)
            logger.debug(traceback.format_exc())
            if attempt <= retries:
                time.sleep(min(2.0 * attempt, 5.0))

    raise RuntimeError(f"Ollama call failed after {retries + 1} attempts: {last_error}")


# -----------------------------------------------------------------------------
# One-shot result checks + deterministic validator
# -----------------------------------------------------------------------------

def one_shot_needs_retry(result: UnifiedPostAnalysis, raw_text: str, expected_media: int) -> list[str]:
    reasons: list[str] = []
    understanding = result.post_understanding

    if not understanding.primary_topics:
        reasons.append("missing_primary_topics")
    if not understanding.post_types:
        reasons.append("missing_post_types")
    if not understanding.intents:
        reasons.append("missing_intents")
    if raw_text.strip() and not understanding.explicit_statements:
        reasons.append("missing_explicit_statements")
    if len(result.media_observations) != expected_media:
        reasons.append(f"media_observation_count:{len(result.media_observations)}!={expected_media}")

    return reasons


def analyze_one_post(
    post: dict,
    provider: str,
    model: str,
    provider_settings: dict,
    output_path: Path,
    logger: logging.Logger = LOGGER,
    show_progress: bool = True,
) -> dict:
    """Analyze one post with exactly one provider call in the normal path."""
    started_total = time.perf_counter()
    post_id = str(post.get("post_id"))
    raw_text = post.get("text") or ""

    images, media_mapping, unavailable = available_media(post, logger)

    if not raw_text.strip() and not images:
        output = {
            "schema_version": SCHEMA_VERSION,
            "analysis_version": ANALYSIS_VERSION,
            "provider": provider,
            "model": model,
            "post_id": post_id,
            "created_at": post.get("created_at"),
            "source_url": post.get("url"),
            "raw_text": raw_text,
            "analysis_status": "SKIPPED_EMPTY",
            "one_shot": True,
            "media_observations": unavailable,
            "post_understanding": None,
            "evidence_signals_raw": [],
            "evidence_signals": [],
            "validator_report": {"reason": "No text and no available supported media"},
            "rejected_inferences": [],
            "usage": {},
            "elapsed_seconds": round(time.perf_counter() - started_total, 3),
        }
        save_json(output, output_path)
        return output

    progress = tqdm(
        total=2,
        desc=f"Post {post_id}",
        unit="step",
        dynamic_ncols=True,
        disable=not show_progress,
        leave=False,
    )

    try:
        # Step 1: one multimodal model call.
        progress.set_description("Model inference")
        retries = max(0, int(provider_settings.get("retries", 1)))
        if provider == "openai":
            result, usage = call_openai(
                model=model,
                post=post,
                images=images,
                media_mapping=media_mapping,
                image_detail=str(provider_settings.get("image_detail", "high")),
                retries=retries,
                logger=logger,
            )
        else:
            result, usage = call_local(
                model=model,
                post=post,
                images=images,
                media_mapping=media_mapping,
                retries=retries,
                logger=logger,
            )
        progress.update(1)

        structural_warnings = one_shot_needs_retry(result, raw_text, len(images))
        if structural_warnings:
            # We do not automatically issue another semantic call here. Provider-level
            # retry is reserved for request/parsing errors. Record quality warnings so
            # the user can inspect them without unexpectedly doubling API cost.
            logger.warning("Structured quality warnings | %s", ", ".join(structural_warnings))

        # Step 2: deterministic normalization / validation.
        progress.set_description("Validator")
        normalized_intents, intent_events = normalize_post_intents(
            raw_text,
            list(result.post_understanding.intents),
        )
        result.post_understanding.intents = normalized_intents

        raw_signal_dicts = [signal.model_dump() for signal in result.evidence_signals]
        kept, report, validator_rejected = validate_signal_dicts(raw_signal_dicts, max_signals=4)
        rejected_all = list(result.rejected_inferences) + validator_rejected

        for event in report.events:
            if event.action == "CORRECT_DIMENSION":
                logger.info(
                    "Validator correction | %s -> %s | %s",
                    event.before_dimension, event.after_dimension, event.claim,
                )
            else:
                logger.debug("Validator %s | %s", event.action, event.reason)

        observations = [
            {"status": "ok", **obs.model_dump()}
            for obs in result.media_observations
        ] + unavailable
        observations.sort(key=lambda x: int(x.get("media_index", 0)))
        progress.update(1)

        output = {
            "schema_version": SCHEMA_VERSION,
            "analysis_version": ANALYSIS_VERSION,
            "provider": provider,
            "model": model,
            "post_id": post_id,
            "created_at": post.get("created_at"),
            "source_url": post.get("url"),
            "raw_text": raw_text,
            "analysis_status": "OK" if not structural_warnings else "OK_WITH_WARNINGS",
            "one_shot": True,
            "media_observations": observations,
            "post_understanding": result.post_understanding.model_dump(),
            "evidence_signals_raw": raw_signal_dicts,
            "evidence_signals": kept,
            "validator_report": {
                **report.as_dict(),
                "intent_events": intent_events,
                "structural_warnings": structural_warnings,
            },
            "rejected_inferences": rejected_all,
            "usage": usage,
            "elapsed_seconds": round(time.perf_counter() - started_total, 3),
        }
        save_json(output, output_path)
        return output
    finally:
        progress.close()


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SelfTrace single-post analyzer v2.0")
    parser.add_argument("--input", default="data/processed/posts_enriched.json", type=Path)
    parser.add_argument("--post-id", default=None)
    parser.add_argument("--config", default="config.json", type=Path)
    parser.add_argument("--provider", choices=["openai", "local"], default=None,
                        help="Override config provider")
    parser.add_argument("--model", default=None, help="Override provider model")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_started = time.perf_counter()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    config = load_config(args.config)
    provider, model, provider_settings = resolve_provider_settings(config, args.provider, args.model)

    input_path = resolve_project_path(args.input)
    data = load_json(input_path)
    post = find_post(data, args.post_id)
    post_id = str(post.get("post_id"))

    if args.output:
        output_path = resolve_project_path(args.output)
    else:
        output_path = (PROJECT_ROOT / "data" / "analysis" / provider / f"{post_id}.json").resolve()

    log_file = (PROJECT_ROOT / "logs" / f"analyze_post_{provider}_{post_id}_{timestamp}.log").resolve()
    setup_logging(log_file, args.log_level)

    LOGGER.info("=" * 82)
    LOGGER.info("SelfTrace analyzer v%s", ANALYSIS_VERSION)
    LOGGER.info("Project root : %s", PROJECT_ROOT)
    LOGGER.info("Provider     : %s", provider)
    LOGGER.info("Model        : %s", model)
    LOGGER.info("Input        : %s", input_path)
    LOGGER.info("Output       : %s", output_path)
    LOGGER.info("Post ID      : %s", post_id)
    LOGGER.info("Media count  : %d", len(post.get("media", [])))
    LOGGER.info("Log file     : %s", log_file)
    LOGGER.info("=" * 82)

    try:
        output = analyze_one_post(
            post=post,
            provider=provider,
            model=model,
            provider_settings=provider_settings,
            output_path=output_path,
            logger=LOGGER,
            show_progress=not args.no_progress,
        )
        LOGGER.info(
            "Analysis completed | status=%s | signals=%d | elapsed=%.2fs",
            output["analysis_status"], len(output["evidence_signals"]), output["elapsed_seconds"],
        )
        for idx, signal in enumerate(output["evidence_signals"], 1):
            LOGGER.info("Signal %d | %s / %s | %s", idx, signal["dimension"], signal["confidence"], signal["claim"])
        LOGGER.info("Saved JSON   : %s", output_path)
        LOGGER.info("Detailed log : %s", log_file)
        LOGGER.info("Total run    : %.2fs", time.perf_counter() - run_started)
    except Exception as exc:
        LOGGER.error("Analysis failed | %s", exc)
        LOGGER.debug(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
