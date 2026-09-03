#!/usr/bin/env python3
"""SelfTrace single-post multimodal analyzer v1.8.

Pipeline
--------
Raw Post + local media
    -> objective media observations
    -> post-level understanding
    -> model-generated narrow evidence signals
    -> deterministic validation/calibration
    -> auditable analysis JSON

All relative paths are resolved from the local-pipeline project root, which is
derived at runtime from this file's location.  The Terminal working directory
therefore does not affect input/output paths.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Literal, Type

from ollama import chat
from pydantic import BaseModel, Field
from tqdm import tqdm

from signal_validator import normalize_post_intents, validate_signal_dicts


# -----------------------------------------------------------------------------
# Version and project paths
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_VERSION = "1.8"
SCHEMA_VERSION = "selftrace-analysis-v1.8"
LOGGER = logging.getLogger("selftrace.analyze_post")


# -----------------------------------------------------------------------------
# Structured schemas
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


class SignalBundle(BaseModel):
    signals: list[EvidenceSignal] = Field(default_factory=list)
    rejected_inferences: list[str] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# Prompts
# -----------------------------------------------------------------------------

MEDIA_SYSTEM = """You are SelfTrace's objective visual-observation layer.
Return structured facts supported by the image only.
Rules:
- Do not infer personality or stable traits.
- Do not infer sensitive attributes.
- For text-heavy screenshots/notes, visible_text MUST contain representative legible text snippets and concepts MUST contain the main concepts.
- Do not put all information only in description.
- NOTE = text note / explanatory slide / bullet-note screenshot.
- PAPER_SCREENSHOT only with clear evidence of an academic paper page.
- TABLE only when content is genuinely organized in rows/columns.
- If text is unreadable, state uncertainty rather than inventing it.
"""

POST_SYSTEM = """You are SelfTrace's post-understanding layer.
Synthesize post text plus media observations.
Rules:
- primary_topics MUST contain 1-5 concrete topics when content is interpretable.
- post_types MUST contain at least one type.
- intents MUST contain at least one intent.
- explicit_statements MUST preserve explicit author claims/preferences/goals when present.
- Separate author words from material merely shown in images.
- Do not infer stable personality or sensitive attributes.
"""

SIGNAL_SYSTEM = """You are SelfTrace's narrow evidence-signal layer.
Generate 0-4 useful signals from THIS POST ONLY.
Dimension semantics:
- INTEREST: explicit attraction, fascination, preference, curiosity toward a topic.
- KNOWLEDGE: demonstrated understanding or technical knowledge, not mere exposure.
- ACTIVITY: studying, reading, implementing, experimenting, documenting work.
- THINKING_PATTERN: a reasoning operation shown here (comparison, abstraction, analogy, mechanism-seeking, pattern-linking), phrased as behavior in this post only.
- EXPRESSION_STYLE: how the author communicates (concise, note-like, bilingual, question-driven), not what they are interested in.
- GOAL_ORIENTATION: explicit future intention, plan, learning/research goal.
Rules:
- HIGH = directly explicit in author text.
- MEDIUM = reasonable inference from behavior/content.
- Do NOT create a signal whose only claim is that the author attached/included an image showing X.
- Do NOT split one coherent idea into near-duplicate signals.
- Do NOT infer stable personality or sensitive attributes.
"""


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

class TqdmLoggingHandler(logging.Handler):
    """Write console logs without destroying tqdm progress bars."""

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
# Paths and JSON
# -----------------------------------------------------------------------------

def resolve_project_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    return p.resolve() if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


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


# -----------------------------------------------------------------------------
# Ollama structured call with retry and timing
# -----------------------------------------------------------------------------

def ollama_json(
    model: str,
    schema_model: Type[BaseModel],
    system: str,
    user: str,
    images: list[Path] | None = None,
    call_name: str = "ollama_call",
    retries: int = 2,
    logger: logging.Logger = LOGGER,
):
    message: dict = {"role": "user", "content": user}
    if images:
        message["images"] = [str(x) for x in images]

    last_error: Exception | None = None
    for attempt in range(1, retries + 2):
        started = time.perf_counter()
        logger.info(
            "Ollama start | %s | attempt=%d/%d | model=%s | images=%d",
            call_name, attempt, retries + 1, model, len(images or []),
        )
        try:
            response = chat(
                model=model,
                format=schema_model.model_json_schema(),
                messages=[{"role": "system", "content": system}, message],
                options={"temperature": 0},
            )
            content = response.message.content
            elapsed = time.perf_counter() - started
            logger.info("Ollama done  | %s | %.2fs | chars=%d", call_name, elapsed, len(content or ""))
            logger.debug("Ollama response | %s | %s", call_name, (content or "")[:12000])
            return schema_model.model_validate_json(content)
        except Exception as exc:
            last_error = exc
            logger.warning("Ollama failed | %s | attempt=%d | %s", call_name, attempt, exc)
            logger.debug(traceback.format_exc())
            if attempt <= retries:
                time.sleep(min(2.0 * attempt, 5.0))

    raise RuntimeError(f"Ollama call failed after {retries + 1} attempts: {call_name}: {last_error}")


# -----------------------------------------------------------------------------
# Media analysis
# -----------------------------------------------------------------------------

def media_needs_retry(obs: MediaObservation) -> bool:
    text_heavy = obs.media_type in {"NOTE", "SCREENSHOT", "PAPER_SCREENSHOT", "TABLE", "CODE", "FORMULA"}
    return text_heavy and (not obs.visible_text or not obs.concepts)


def analyze_media(model: str, path: Path, post_text: str, media_index: int, retries: int, logger: logging.Logger) -> MediaObservation:
    prompt = f"""Analyze this image as evidence attached to a social-media post.
Post text for context (do not let it override the pixels):
{post_text or '[empty]'}

Return objective visual observations. Preserve legible technical terms. For text-heavy material, populate visible_text and concepts."""
    obs = ollama_json(
        model, MediaObservation, MEDIA_SYSTEM, prompt,
        images=[path], call_name=f"media#{media_index}:initial", retries=retries, logger=logger,
    )
    if media_needs_retry(obs):
        logger.warning("Media under-filled; structured retry | media_index=%s", media_index)
        retry_prompt = f"""The previous structured result was under-filled:
{json.dumps(obs.model_dump(), ensure_ascii=False)}
Re-analyze the same image. For text-heavy content, visible_text and concepts must be populated with legible evidence. Do not invent unreadable content."""
        obs = ollama_json(
            model, MediaObservation, MEDIA_SYSTEM, retry_prompt,
            images=[path], call_name=f"media#{media_index}:structured_retry", retries=retries, logger=logger,
        )
    return obs


# -----------------------------------------------------------------------------
# Post understanding + deterministic intent normalization
# -----------------------------------------------------------------------------

def post_needs_retry(understanding: PostUnderstanding, raw_text: str) -> bool:
    if not understanding.primary_topics or not understanding.post_types or not understanding.intents:
        return True
    return bool(raw_text.strip()) and not understanding.explicit_statements


def analyze_post_understanding(model: str, post: dict, media_obs: list[dict], retries: int, logger: logging.Logger) -> tuple[PostUnderstanding, list[dict]]:
    payload = {
        "post_id": str(post.get("post_id")),
        "created_at": post.get("created_at"),
        "text": post.get("text") or "",
        "media_observations": media_obs,
    }
    prompt = "Synthesize this post into the required structured fields:\n" + json.dumps(payload, ensure_ascii=False)
    understanding = ollama_json(
        model, PostUnderstanding, POST_SYSTEM, prompt,
        call_name="post_understanding:initial", retries=retries, logger=logger,
    )

    if post_needs_retry(understanding, payload["text"]):
        logger.warning(
            "Post understanding incomplete; retrying | topics=%d types=%d intents=%d explicit=%d",
            len(understanding.primary_topics), len(understanding.post_types),
            len(understanding.intents), len(understanding.explicit_statements),
        )
        retry_prompt = f"""The previous output was incomplete:
{json.dumps(understanding.model_dump(), ensure_ascii=False)}
Return a corrected result. For interpretable content, primary_topics, post_types and intents must be non-empty. If raw author text contains explicit preferences/goals/claims, capture them in explicit_statements."""
        understanding = ollama_json(
            model, PostUnderstanding, POST_SYSTEM, retry_prompt,
            call_name="post_understanding:retry", retries=retries, logger=logger,
        )

    normalized_intents, intent_events = normalize_post_intents(payload["text"], list(understanding.intents))
    understanding.intents = normalized_intents
    return understanding, intent_events


# -----------------------------------------------------------------------------
# Evidence signals + deterministic validation
# -----------------------------------------------------------------------------

def extract_signals(model: str, post: dict, understanding: PostUnderstanding, retries: int, logger: logging.Logger):
    payload = {
        "post_id": str(post.get("post_id")),
        "raw_text": post.get("text") or "",
        "post_understanding": understanding.model_dump(),
    }
    bundle = ollama_json(
        model, SignalBundle, SIGNAL_SYSTEM,
        "Create narrow evidence signals for this post:\n" + json.dumps(payload, ensure_ascii=False),
        call_name="evidence_signals", retries=retries, logger=logger,
    )

    raw_signal_dicts = [signal.model_dump() for signal in bundle.signals]
    kept, report, rejected = validate_signal_dicts(raw_signal_dicts, max_signals=4)

    for event in report.events:
        if event.action == "CORRECT_DIMENSION":
            logger.info(
                "Validator correction | %s -> %s | %s",
                event.before_dimension, event.after_dimension, event.claim,
            )
        else:
            logger.debug("Validator %s | %s", event.action, event.reason)

    rejected_all = list(bundle.rejected_inferences) + rejected
    return kept, rejected_all, report.as_dict(), raw_signal_dicts


# -----------------------------------------------------------------------------
# Reusable one-post pipeline (used by single CLI and batch CLI)
# -----------------------------------------------------------------------------

def analyze_one_post(
    post: dict,
    model: str,
    output_path: Path,
    logger: logging.Logger = LOGGER,
    show_progress: bool = True,
    ollama_retries: int = 2,
) -> dict:
    post_id = str(post.get("post_id"))
    raw_text = post.get("text") or ""
    media_items = post.get("media", [])

    # Empty text + no media is not analyzable; record it rather than hallucinate.
    if not raw_text.strip() and not media_items:
        output = {
            "schema_version": SCHEMA_VERSION,
            "analysis_version": ANALYSIS_VERSION,
            "model": model,
            "post_id": post_id,
            "created_at": post.get("created_at"),
            "source_url": post.get("url"),
            "raw_text": raw_text,
            "analysis_status": "SKIPPED_EMPTY",
            "media_observations": [],
            "post_understanding": None,
            "evidence_signals_raw": [],
            "evidence_signals": [],
            "validator_report": {"reason": "No text and no supported media"},
            "rejected_inferences": [],
        }
        save_json(output, output_path)
        return output

    total_steps = len(media_items) + 2
    progress = tqdm(
        total=total_steps,
        desc=f"Post {post_id}",
        unit="step",
        dynamic_ncols=True,
        disable=not show_progress,
        leave=False,
    )

    observations: list[dict] = []
    try:
        # Stage A: each local image independently.
        for position, media in enumerate(media_items, start=1):
            media_index = int(media.get("index", position - 1))
            progress.set_description(f"Media {position}/{len(media_items)}")
            local_path = media.get("local_path")
            logger.info(
                "Media start | post=%s | %d/%d | media_index=%s | path=%s",
                post_id, position, len(media_items), media_index, local_path,
            )

            if not local_path or media.get("download_status") == "failed":
                reason = media.get("download_error") or "missing local_path"
                logger.warning("Media unavailable | post=%s | index=%s | %s", post_id, media_index, reason)
                observations.append({"media_index": media_index, "status": "unavailable", "reason": reason})
                progress.update(1)
                continue

            path = resolve_project_path(local_path)
            if not path.exists():
                reason = f"file not found: {path}"
                logger.warning("Media unavailable | post=%s | index=%s | %s", post_id, media_index, reason)
                observations.append({"media_index": media_index, "status": "unavailable", "reason": reason})
                progress.update(1)
                continue

            started = time.perf_counter()
            obs = analyze_media(model, path, raw_text, media_index, ollama_retries, logger)
            observations.append({"media_index": media_index, "status": "ok", **obs.model_dump()})
            logger.info(
                "Media done | post=%s | index=%s | type=%s | text=%d | concepts=%d | %.2fs",
                post_id, media_index, obs.media_type, len(obs.visible_text), len(obs.concepts),
                time.perf_counter() - started,
            )
            progress.update(1)

        # Stage B: synthesize the whole post.
        progress.set_description("Post understanding")
        logger.info("Stage start | post=%s | Post Understanding", post_id)
        understanding, intent_events = analyze_post_understanding(model, post, observations, ollama_retries, logger)
        logger.info(
            "Stage done | post=%s | topics=%s | types=%s | intents=%s",
            post_id, understanding.primary_topics, understanding.post_types, understanding.intents,
        )
        progress.update(1)

        # Stage C: generate narrow signals and validate them deterministically.
        progress.set_description("Evidence signals")
        logger.info("Stage start | post=%s | Evidence Signals", post_id)
        signals, rejected, validator_report, raw_signals = extract_signals(
            model, post, understanding, ollama_retries, logger
        )
        validator_report["intent_events"] = intent_events
        logger.info(
            "Stage done | post=%s | raw_signals=%d | kept=%d | corrected=%d | filtered=%d | dup=%d",
            post_id,
            validator_report.get("input_count", 0),
            validator_report.get("kept_count", 0),
            validator_report.get("corrected_count", 0),
            validator_report.get("filtered_count", 0),
            validator_report.get("duplicate_count", 0),
        )
        progress.update(1)

        output = {
            "schema_version": SCHEMA_VERSION,
            "analysis_version": ANALYSIS_VERSION,
            "model": model,
            "post_id": post_id,
            "created_at": post.get("created_at"),
            "source_url": post.get("url"),
            "raw_text": raw_text,
            "analysis_status": "OK",
            "media_observations": observations,
            "post_understanding": understanding.model_dump(),
            # Preserve raw model output for audit/debug; final aggregation uses validated signals below.
            "evidence_signals_raw": raw_signals,
            "evidence_signals": signals,
            "validator_report": validator_report,
            "rejected_inferences": rejected,
        }
        save_json(output, output_path)
        return output
    finally:
        progress.close()


# -----------------------------------------------------------------------------
# Single-post CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SelfTrace single-post multimodal analyzer v1.8")
    parser.add_argument("--input", required=True, type=Path, help="posts_enriched.json")
    parser.add_argument("--post-id", default=None, help="Target post ID")
    parser.add_argument("--model", default="qwen3-vl:8b", help="Ollama model")
    parser.add_argument("--output", type=Path, default=None, help="Optional output JSON path")
    parser.add_argument("--ollama-retries", type=int, default=2, help="Retries per Ollama structured call")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_started = time.perf_counter()
    input_path = resolve_project_path(args.input)
    data = load_json(input_path)
    post = find_post(data, args.post_id)
    post_id = str(post.get("post_id"))

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = resolve_project_path(args.log_file) if args.log_file else (
        PROJECT_ROOT / "logs" / f"analyze_post_{post_id}_{timestamp}.log"
    ).resolve()
    setup_logging(log_file, args.log_level)

    output_path = resolve_project_path(args.output) if args.output else (
        PROJECT_ROOT / "data" / "analysis" / f"{post_id}.json"
    ).resolve()

    LOGGER.info("=" * 78)
    LOGGER.info("SelfTrace analyzer v%s", ANALYSIS_VERSION)
    LOGGER.info("Project root : %s", PROJECT_ROOT)
    LOGGER.info("Input        : %s", input_path)
    LOGGER.info("Output       : %s", output_path)
    LOGGER.info("Post ID      : %s", post_id)
    LOGGER.info("Model        : %s", args.model)
    LOGGER.info("Media count  : %d", len(post.get("media", [])))
    LOGGER.info("Log file     : %s", log_file)
    LOGGER.info("=" * 78)

    try:
        output = analyze_one_post(
            post,
            args.model,
            output_path,
            logger=LOGGER,
            show_progress=not args.no_progress,
            ollama_retries=max(0, args.ollama_retries),
        )
        LOGGER.info("Analysis completed | status=%s | signals=%d", output["analysis_status"], len(output["evidence_signals"]))
        for idx, signal in enumerate(output["evidence_signals"], 1):
            LOGGER.info("Signal %d | %s / %s | %s", idx, signal["dimension"], signal["confidence"], signal["claim"])
        LOGGER.info("Saved JSON   : %s", output_path)
        LOGGER.info("Detailed log : %s", log_file)
        LOGGER.info("Elapsed      : %.2fs", time.perf_counter() - run_started)
    except Exception as exc:
        LOGGER.error("Analysis failed | %s", exc)
        LOGGER.debug(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
