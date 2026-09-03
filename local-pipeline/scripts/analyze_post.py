#!/usr/bin/env python3
"""SelfTrace single-post multimodal analyzer v1.7.

核心目标：
1. 从 posts_enriched.json 读取一条 Post；
2. 使用本地 Ollama 视觉模型逐张分析图片；
3. 合成 Post-level understanding；
4. 提取窄范围、可追溯的 Evidence Signals；
5. 全程提供进度条、实时终端日志和文件日志。

路径约定：
- 所有相对路径都以 local-pipeline 项目根目录为基准；
- 项目根目录通过当前脚本位置动态计算，不依赖 Terminal 当前工作目录。
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Literal, Type

from ollama import chat
from pydantic import BaseModel, Field
from tqdm import tqdm

# -----------------------------------------------------------------------------
# 版本与项目路径
# -----------------------------------------------------------------------------

# scripts/analyze_post.py -> parents[1] 即 local-pipeline 根目录。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_VERSION = "1.7"
SCHEMA_VERSION = "selftrace-analysis-v1.7"

# 日志对象在 main() 中初始化。
LOGGER = logging.getLogger("selftrace.analyze_post")


# -----------------------------------------------------------------------------
# Schema 枚举
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


# -----------------------------------------------------------------------------
# Pydantic 输出结构
# -----------------------------------------------------------------------------

class ChartObservation(BaseModel):
    """图表类媒体的可选结构化信息。"""

    title: str | None = None
    chart_type: str | None = None
    x_axis: str | None = None
    y_axis: str | None = None
    entities: list[str] = Field(default_factory=list)
    main_pattern: str | None = None
    notable_values: list[str] = Field(default_factory=list)


class MediaObservation(BaseModel):
    """单张图片的客观观察结果。"""

    media_type: MEDIA_TYPES
    description: str
    visible_text: list[str] = Field(default_factory=list)
    concepts: list[str] = Field(default_factory=list)
    visual_elements: list[str] = Field(default_factory=list)
    chart: ChartObservation | None = None
    uncertainty: list[str] = Field(default_factory=list)


class PostUnderstanding(BaseModel):
    """文字 + 全部图片之后的 Post-level 综合理解。"""

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
    """单条 Post 支持的窄范围 Evidence Signal。"""

    dimension: DIMENSIONS
    claim: str
    evidence: list[str]
    confidence: CONFIDENCE
    temporal_scope: Literal["THIS_POST_ONLY"] = "THIS_POST_ONLY"


class SignalBundle(BaseModel):
    """保留的 signals 与被拒绝/过滤的推断。"""

    signals: list[EvidenceSignal] = Field(default_factory=list)
    rejected_inferences: list[str] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# Prompt
# -----------------------------------------------------------------------------

MEDIA_SYSTEM = """You are SelfTrace's objective visual-observation layer.
Return structured facts supported by the image only.
Rules:
- Do not infer personality or stable traits.
- Do not infer sensitive attributes.
- For text-heavy screenshots/notes, visible_text MUST contain representative legible text snippets and concepts MUST contain the main technical/semantic concepts.
- Do not put all information only in description.
- NOTE = text note / explanatory slide / bullet-note screenshot.
- PAPER_SCREENSHOT only when there is clear evidence of an academic paper page (paper title/authors/abstract/section/figure/citation formatting).
- TABLE only when content is genuinely organized in rows/columns.
- If text is unreadable, state that in uncertainty rather than inventing it.
"""

POST_SYSTEM = """You are SelfTrace's post-understanding layer.
Synthesize post text plus media observations.
Rules:
- primary_topics MUST contain 1-5 concrete topics when content is interpretable.
- post_types MUST contain at least one type.
- intents MUST contain at least one intent.
- explicit_statements MUST preserve the author's explicit claims/preferences/goals from the post text when present.
- Separate the author's own words from material merely shown in images.
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
- Do NOT split one coherent idea into many near-duplicate signals.
- Do NOT infer stable personality, diagnosis, politics, religion, sexuality, health, ethnicity, or relationships.
"""


# -----------------------------------------------------------------------------
# 日志基础设施
# -----------------------------------------------------------------------------

class TqdmLoggingHandler(logging.Handler):
    """使用 tqdm.write 输出日志，避免日志把进度条冲乱。"""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            tqdm.write(msg, file=sys.stderr)
        except Exception:
            self.handleError(record)


def setup_logging(log_file: Path, console_level: str = "INFO") -> None:
    """同时配置终端日志和详细文件日志。"""

    log_file.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.setLevel(logging.DEBUG)
    LOGGER.handlers.clear()
    LOGGER.propagate = False

    # 终端：默认 INFO，显示用户关心的实时阶段信息。
    console_handler = TqdmLoggingHandler()
    console_handler.setLevel(getattr(logging, console_level.upper(), logging.INFO))
    console_handler.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))

    # 文件：固定 DEBUG，保存完整时间戳、函数与行号，便于排障。
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(funcName)s:%(lineno)d | %(message)s"
    ))

    LOGGER.addHandler(console_handler)
    LOGGER.addHandler(file_handler)


# -----------------------------------------------------------------------------
# 通用路径 / JSON 工具
# -----------------------------------------------------------------------------


def resolve_project_path(path: str | Path) -> Path:
    """把相对路径解释为 PROJECT_ROOT 下的路径。"""

    p = Path(path).expanduser()
    return p.resolve() if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def load_json(path: Path) -> dict:
    """读取 UTF-8 JSON。"""

    LOGGER.debug("Loading JSON: %s", path)
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(obj: dict, path: Path) -> None:
    """保存格式化 UTF-8 JSON。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.debug("JSON saved: %s (%d bytes)", path, path.stat().st_size)


def find_post(data: dict, post_id: str | None) -> dict:
    """按 post_id 找 Post；未指定时选第一条包含 media 的 Post。"""

    posts = data.get("posts", [])
    if post_id:
        for post in posts:
            if str(post.get("post_id")) == str(post_id):
                return post
        raise SystemExit(f"Post not found: {post_id}")

    for post in posts:
        if post.get("media"):
            return post
    raise SystemExit("No post with media found")


# -----------------------------------------------------------------------------
# Ollama 调用封装
# -----------------------------------------------------------------------------


def ollama_json(
    model: str,
    schema_model: Type[BaseModel],
    system: str,
    user: str,
    images: list[Path] | None = None,
    call_name: str = "ollama_call",
):
    """调用 Ollama 并将 JSON 响应直接校验为 Pydantic model。

    日志记录：调用名称、模型、图片数量、耗时、响应长度。
    DEBUG 日志还保存截断后的原始响应，便于定位结构化输出问题。
    """

    message: dict = {"role": "user", "content": user}
    if images:
        message["images"] = [str(x) for x in images]

    LOGGER.info("Ollama start | %s | model=%s | images=%d", call_name, model, len(images or []))
    start = time.perf_counter()

    try:
        response = chat(
            model=model,
            format=schema_model.model_json_schema(),
            messages=[{"role": "system", "content": system}, message],
            options={"temperature": 0},
        )
        elapsed = time.perf_counter() - start
        raw = response.message.content
        LOGGER.info("Ollama done  | %s | %.2fs | response_chars=%d", call_name, elapsed, len(raw or ""))
        LOGGER.debug("Ollama raw response | %s | %s", call_name, (raw or "")[:4000])
        return schema_model.model_validate_json(raw)
    except Exception:
        elapsed = time.perf_counter() - start
        LOGGER.exception("Ollama failed | %s | after %.2fs", call_name, elapsed)
        raise


# -----------------------------------------------------------------------------
# Media Observation
# -----------------------------------------------------------------------------


def media_needs_retry(obs: MediaObservation) -> bool:
    """判断文字型图片是否出现字段填充不足。"""

    text_heavy = obs.media_type in {
        "SCREENSHOT", "PAPER_SCREENSHOT", "NOTE", "TABLE", "DIAGRAM", "CODE", "FORMULA", "OTHER"
    }
    return text_heavy and len(obs.description) > 80 and (not obs.visible_text or not obs.concepts)


def analyze_media(model: str, path: Path, post_text: str, media_index: int) -> MediaObservation:
    """分析单张图片；必要时自动重试一次以补全 visible_text / concepts。"""

    prompt = f"""Analyze this image attached to a social-media post.
Post text for context only:
{post_text or '[empty]'}

Populate every applicable field. For a text-heavy image, extract representative visible_text and concepts instead of leaving them empty."""

    obs = ollama_json(
        model,
        MediaObservation,
        MEDIA_SYSTEM,
        prompt,
        [path],
        call_name=f"media#{media_index}:initial",
    )

    if media_needs_retry(obs):
        LOGGER.warning(
            "Media #%s under-filled structured fields; retrying | type=%s visible_text=%d concepts=%d",
            media_index, obs.media_type, len(obs.visible_text), len(obs.concepts)
        )
        retry_prompt = f"""Your previous structured output under-filled visible_text/concepts.
Previous output:
{json.dumps(obs.model_dump(), ensure_ascii=False)}

Re-analyze the SAME image. Correct media_type if needed. For any legible text-heavy image, visible_text and concepts must be populated with concise representative items."""
        obs = ollama_json(
            model,
            MediaObservation,
            MEDIA_SYSTEM,
            retry_prompt,
            [path],
            call_name=f"media#{media_index}:retry",
        )

    return obs


# -----------------------------------------------------------------------------
# Post Understanding
# -----------------------------------------------------------------------------


def post_needs_retry(understanding: PostUnderstanding, raw_text: str) -> bool:
    """检查 Post-level 关键字段是否被模型留空。"""

    if not understanding.primary_topics or not understanding.post_types or not understanding.intents:
        return True
    if raw_text.strip() and not understanding.explicit_statements:
        return True
    return False


def analyze_post_understanding(model: str, post: dict, media_obs: list[dict]) -> PostUnderstanding:
    """综合原文与所有图片观察，形成 Post-level understanding。"""

    payload = {
        "post_id": str(post.get("post_id")),
        "created_at": post.get("created_at"),
        "text": post.get("text") or "",
        "media_observations": media_obs,
    }
    user_prompt = "Synthesize this post into the required structured fields:\n" + json.dumps(payload, ensure_ascii=False)

    understanding = ollama_json(
        model,
        PostUnderstanding,
        POST_SYSTEM,
        user_prompt,
        call_name="post_understanding:initial",
    )

    if post_needs_retry(understanding, payload["text"]):
        LOGGER.warning(
            "Post understanding incomplete; retrying | topics=%d post_types=%d intents=%d explicit=%d",
            len(understanding.primary_topics), len(understanding.post_types),
            len(understanding.intents), len(understanding.explicit_statements)
        )
        retry_prompt = f"""The previous output was incomplete.
Previous output:
{json.dumps(understanding.model_dump(), ensure_ascii=False)}

Return a corrected output. primary_topics, post_types, and intents must be non-empty for interpretable content. Because the raw post text is non-empty, explicit_statements must capture the author's explicit statements/preferences/goals."""
        understanding = ollama_json(
            model,
            PostUnderstanding,
            POST_SYSTEM,
            retry_prompt,
            call_name="post_understanding:retry",
        )

    return understanding


# -----------------------------------------------------------------------------
# Evidence Signal
# -----------------------------------------------------------------------------


def is_media_only_signal(signal: EvidenceSignal) -> bool:
    """过滤仅描述“作者附了一张图”的伪 Signal。"""

    text = (signal.claim + " " + " ".join(signal.evidence)).lower()
    patterns = [
        "included media", "attached media", "attached image", "included image",
        "media showing", "image showing", "screenshot showing"
    ]
    return any(pattern in text for pattern in patterns)


def semantic_dimension(signal: EvidenceSignal) -> str:
    """用轻量规则纠正模型非常明显的维度误分类。"""

    text = (signal.claim + " " + " ".join(signal.evidence)).lower()
    if re.search(r"fascinat|interest|interested|curios|drawn to|appeal", text):
        return "INTEREST"
    if re.search(r"would like|plan|intend|future|goal|systematically study|want to study|aim", text):
        return "GOAL_ORIENTATION"
    if re.search(r"implement|experiment|studying|study of|learning|reading|documenting|working on|engag", text):
        return "ACTIVITY"
    if re.search(r"compare|comparison|abstract|analogy|same core|shared core|mechanism|pattern|connects .* concepts", text):
        return "THINKING_PATTERN"
    if re.search(r"demonstrates understanding|understands|explains|technical knowledge|knowledge of", text):
        return "KNOWLEDGE"
    return signal.dimension


def normalize_signals(bundle: SignalBundle) -> SignalBundle:
    """过滤伪 Signal、修正明显维度、去重并限制最多 4 条。"""

    kept: list[EvidenceSignal] = []
    seen: set[tuple[str, str]] = set()
    rejected = list(bundle.rejected_inferences)

    for signal in bundle.signals:
        if is_media_only_signal(signal):
            reason = f"Filtered non-signal media-description claim: {signal.claim}"
            LOGGER.debug(reason)
            rejected.append(reason)
            continue

        original_dimension = signal.dimension
        signal.dimension = semantic_dimension(signal)
        if signal.dimension != original_dimension:
            LOGGER.info(
                "Signal dimension corrected | %s -> %s | %s",
                original_dimension, signal.dimension, signal.claim
            )

        key = (signal.dimension, re.sub(r"\W+", " ", signal.claim.lower()).strip())
        if key in seen:
            LOGGER.debug("Duplicate signal filtered: %s", signal.claim)
            continue
        seen.add(key)
        kept.append(signal)

    # HIGH 优先，最多保留 4 条，避免单条 Post 过度展开。
    kept.sort(key=lambda item: 0 if item.confidence == "HIGH" else 1)
    return SignalBundle(signals=kept[:4], rejected_inferences=rejected)


def extract_signals(model: str, post: dict, understanding: PostUnderstanding) -> SignalBundle:
    """生成并规范化 Evidence Signals。"""

    payload = {
        "post_id": str(post.get("post_id")),
        "raw_text": post.get("text") or "",
        "post_understanding": understanding.model_dump(),
    }
    bundle = ollama_json(
        model,
        SignalBundle,
        SIGNAL_SYSTEM,
        "Create narrow evidence signals for this post:\n" + json.dumps(payload, ensure_ascii=False),
        call_name="evidence_signals",
    )
    return normalize_signals(bundle)


# -----------------------------------------------------------------------------
# CLI / Main
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="SelfTrace single-post multimodal analyzer v1.7")
    parser.add_argument("--input", required=True, type=Path, help="posts_enriched.json")
    parser.add_argument("--post-id", default=None, help="目标 Post ID；不传则选择第一条带 media 的 Post")
    parser.add_argument("--model", default="qwen3-vl:8b", help="Ollama 模型名")
    parser.add_argument("--output", type=Path, default=None, help="可选输出 JSON 路径")
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="终端日志级别；日志文件始终保存 DEBUG 级别"
    )
    parser.add_argument("--log-file", type=Path, default=None, help="可选日志文件路径")
    parser.add_argument("--no-progress", action="store_true", help="关闭进度条（适合 CI/重定向输出）")
    return parser.parse_args()


def main() -> None:
    """单条 Post 分析主流程。"""

    args = parse_args()
    run_start = time.perf_counter()

    input_path = resolve_project_path(args.input)
    data = load_json(input_path) if LOGGER.handlers else json.loads(input_path.read_text(encoding="utf-8"))
    post = find_post(data, args.post_id)
    post_id = str(post.get("post_id"))

    # 默认日志名包含 Post ID 与时间，避免每次运行覆盖上一份日志。
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    default_log = PROJECT_ROOT / "logs" / f"analyze_post_{post_id}_{timestamp}.log"
    log_file = resolve_project_path(args.log_file) if args.log_file else default_log.resolve()
    setup_logging(log_file, args.log_level)

    # 日志初始化后重新记录关键输入信息。
    LOGGER.info("=" * 72)
    LOGGER.info("SelfTrace analyzer v%s", ANALYSIS_VERSION)
    LOGGER.info("Project root : %s", PROJECT_ROOT)
    LOGGER.info("Input        : %s", input_path)
    LOGGER.info("Post ID      : %s", post_id)
    LOGGER.info("Model        : %s", args.model)
    LOGGER.info("Media count  : %d", len(post.get("media", [])))
    LOGGER.info("Log file     : %s", log_file)
    LOGGER.info("=" * 72)

    observations: list[dict] = []
    media_items = post.get("media", [])

    # 总进度 = 每张图片 + Post Understanding + Evidence Signals。
    total_steps = len(media_items) + 2
    progress = tqdm(
        total=total_steps,
        desc="SelfTrace",
        unit="step",
        dynamic_ncols=True,
        disable=args.no_progress,
    )

    try:
        # ------------------------------------------------------------------
        # Stage 1: Media observations
        # ------------------------------------------------------------------
        for position, media in enumerate(media_items, start=1):
            media_index = int(media.get("index", position - 1))
            progress.set_description(f"Media {position}/{len(media_items)}")

            local_path = media.get("local_path")
            LOGGER.info(
                "Media start | %d/%d | media_index=%s | local_path=%s",
                position, len(media_items), media_index, local_path
            )

            if not local_path or media.get("download_status") == "failed":
                reason = media.get("download_error") or "missing local_path"
                LOGGER.warning("Media unavailable | media_index=%s | %s", media_index, reason)
                observations.append({
                    "media_index": media_index,
                    "status": "unavailable",
                    "reason": reason,
                })
                progress.update(1)
                continue

            path = resolve_project_path(local_path)
            if not path.exists():
                reason = f"file not found: {path}"
                LOGGER.warning("Media unavailable | media_index=%s | %s", media_index, reason)
                observations.append({
                    "media_index": media_index,
                    "status": "unavailable",
                    "reason": reason,
                })
                progress.update(1)
                continue

            media_start = time.perf_counter()
            obs = analyze_media(args.model, path, post.get("text") or "", media_index)
            elapsed = time.perf_counter() - media_start
            observations.append({"media_index": media_index, "status": "ok", **obs.model_dump()})

            LOGGER.info(
                "Media done  | media_index=%s | type=%s | concepts=%d | visible_text=%d | %.2fs",
                media_index, obs.media_type, len(obs.concepts), len(obs.visible_text), elapsed
            )
            progress.update(1)

        # ------------------------------------------------------------------
        # Stage 2: Post-level understanding
        # ------------------------------------------------------------------
        progress.set_description("Post understanding")
        LOGGER.info("Stage start | Post Understanding")
        stage_start = time.perf_counter()
        understanding = analyze_post_understanding(args.model, post, observations)
        LOGGER.info(
            "Stage done  | Post Understanding | topics=%s | post_types=%s | intents=%s | %.2fs",
            understanding.primary_topics,
            understanding.post_types,
            understanding.intents,
            time.perf_counter() - stage_start,
        )
        progress.update(1)

        # ------------------------------------------------------------------
        # Stage 3: Evidence Signal extraction
        # ------------------------------------------------------------------
        progress.set_description("Evidence signals")
        LOGGER.info("Stage start | Evidence Signals")
        stage_start = time.perf_counter()
        signals = extract_signals(args.model, post, understanding)
        LOGGER.info(
            "Stage done  | Evidence Signals | kept=%d rejected=%d | %.2fs",
            len(signals.signals), len(signals.rejected_inferences), time.perf_counter() - stage_start
        )
        progress.update(1)

        # ------------------------------------------------------------------
        # Stage 4: Save final output
        # ------------------------------------------------------------------
        output = {
            "schema_version": SCHEMA_VERSION,
            "analysis_version": ANALYSIS_VERSION,
            "model": args.model,
            "post_id": post_id,
            "created_at": post.get("created_at"),
            "source_url": post.get("url"),
            "raw_text": post.get("text") or "",
            "media_observations": observations,
            "post_understanding": understanding.model_dump(),
            "evidence_signals": [signal.model_dump() for signal in signals.signals],
            "rejected_inferences": signals.rejected_inferences,
        }

        output_path = (
            resolve_project_path(args.output)
            if args.output
            else (PROJECT_ROOT / "data" / "analysis" / f"{post_id}.json").resolve()
        )
        save_json(output, output_path)

        # ------------------------------------------------------------------
        # Human-readable summary
        # ------------------------------------------------------------------
        total_elapsed = time.perf_counter() - run_start
        LOGGER.info("-" * 72)
        LOGGER.info("Analysis completed successfully")
        LOGGER.info("Signals kept  : %d", len(signals.signals))
        for number, signal in enumerate(signals.signals, start=1):
            LOGGER.info(
                "Signal %d      : %s / %s | %s",
                number, signal.dimension, signal.confidence, signal.claim
            )
        LOGGER.info("Rejected      : %d", len(signals.rejected_inferences))
        LOGGER.info("Output JSON   : %s", output_path)
        LOGGER.info("Detailed log  : %s", log_file)
        LOGGER.info("Total elapsed : %.2fs", total_elapsed)
        LOGGER.info("-" * 72)

    except KeyboardInterrupt:
        LOGGER.warning("Interrupted by user (Ctrl+C).")
        raise SystemExit(130)
    except Exception as exc:
        LOGGER.error("Analysis failed: %s", exc)
        LOGGER.debug("Traceback:\n%s", traceback.format_exc())
        raise
    finally:
        progress.close()


if __name__ == "__main__":
    main()
