#!/usr/bin/env python3
import argparse
import hashlib
import json
import mimetypes
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import requests
from PIL import Image

# Always resolve project-relative paths from local-pipeline/, not from cwd.
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_project_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (PROJECT_ROOT / p).resolve()


def path_for_json(path: Path) -> str:
    """Store paths relative to project root when possible, otherwise absolute."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def set_quality(url: str, quality: str) -> str:
    p = urlparse(url)
    q = parse_qs(p.query)
    q["name"] = [quality]
    return urlunparse((p.scheme, p.netloc, p.path, p.params, urlencode(q, doseq=True), p.fragment))


def candidates(url: str) -> list[str]:
    out = [set_quality(url, "orig"), set_quality(url, "large"), url]
    seen = set()
    result = []
    for u in out:
        if u not in seen:
            seen.add(u)
            result.append(u)
    return result


def extension_from_url(url: str) -> str:
    q = parse_qs(urlparse(url).query)
    fmt = q.get("format", [""])[0].lower()
    if fmt == "jpeg":
        fmt = "jpg"
    if fmt in {"jpg", "png", "webp", "gif"}:
        return fmt
    return "jpg"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_image(path: Path) -> dict:
    try:
        with Image.open(path) as im:
            width, height = im.size
            fmt = im.format
            im.verify()
        return {"valid": True, "width": width, "height": height, "format": fmt, "error": None}
    except Exception as e:
        return {"valid": False, "width": None, "height": None, "format": None, "error": str(e)}


def download_one(url: str, dest: Path, timeout: int = 25) -> dict:
    temp = dest.with_suffix(dest.suffix + ".part")
    try:
        with requests.get(
            url,
            timeout=timeout,
            stream=True,
            headers={"User-Agent": "Mozilla/5.0 SelfTrace/0.1"},
        ) as r:
            r.raise_for_status()
            content_type = r.headers.get("Content-Type", "").split(";")[0].strip()
            if content_type and not content_type.startswith("image/"):
                raise ValueError(f"Unexpected Content-Type: {content_type}")
            with temp.open("wb") as f:
                for chunk in r.iter_content(64 * 1024):
                    if chunk:
                        f.write(chunk)
        if not temp.exists() or temp.stat().st_size == 0:
            raise ValueError("Empty download")
        temp.replace(dest)
        return {"ok": True, "content_type": content_type or None, "error": None}
    except Exception as e:
        temp.unlink(missing_ok=True)
        return {"ok": False, "content_type": None, "error": str(e)}


def infer_content_type(post: dict) -> str:
    has_text = bool((post.get("text") or "").strip())
    has_media = bool(post.get("media"))
    if has_text and has_media:
        return "TEXT_IMAGE"
    if has_text:
        return "TEXT"
    if has_media:
        return "IMAGE"
    return "UNKNOWN"


def process_media(post: dict, media: dict, media_dir: Path) -> dict:
    post_id = str(post["post_id"])
    idx = int(media.get("index", 0))
    original_url = media.get("src") or media.get("original_url")
    if not original_url:
        return {**media, "download_status": "failed", "download_error": "missing source URL"}

    ext = extension_from_url(original_url)
    dest = (media_dir / f"{post_id}_{idx}.{ext}").resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)

    result = {
        **media,
        "original_url": original_url,
        "resolved_url": None,
        # IMPORTANT: stable project-relative path, e.g. data/media/123_0.jpg
        "local_path": path_for_json(dest),
        "file_name": dest.name,
        "mime_type": None,
        "width": None,
        "height": None,
        "sha256": None,
        "download_status": None,
        "download_error": None,
    }

    if dest.exists():
        v = validate_image(dest)
        if v["valid"]:
            result.update(
                download_status="exists",
                width=v["width"],
                height=v["height"],
                sha256=sha256_file(dest),
                mime_type=mimetypes.guess_type(dest.name)[0],
            )
            return result
        dest.unlink(missing_ok=True)

    last_error = None
    for candidate in candidates(original_url):
        d = download_one(candidate, dest)
        if not d["ok"]:
            last_error = d["error"]
            continue
        v = validate_image(dest)
        if not v["valid"]:
            last_error = f"invalid image: {v['error']}"
            dest.unlink(missing_ok=True)
            continue
        result.update(
            resolved_url=candidate,
            download_status="downloaded",
            width=v["width"],
            height=v["height"],
            sha256=sha256_file(dest),
            mime_type=d["content_type"] or mimetypes.guess_type(dest.name)[0],
        )
        return result

    result["download_status"] = "failed"
    result["download_error"] = last_error or "all candidate URLs failed"
    return result


def main():
    ap = argparse.ArgumentParser(description="Download SelfTrace X media to local disk")
    ap.add_argument("--input", default="data/raw/x_posts.json")
    ap.add_argument("--media-dir", default="data/media")
    ap.add_argument("--output-posts", default="data/processed/posts_enriched.json")
    ap.add_argument("--output-manifest", default="data/processed/media_manifest.json")
    args = ap.parse_args()

    input_path = resolve_project_path(args.input)
    media_dir = resolve_project_path(args.media_dir)
    output_posts = resolve_project_path(args.output_posts)
    output_manifest = resolve_project_path(args.output_manifest)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Input:        {input_path}")
    print(f"Media dir:    {media_dir}")

    data = load_json(input_path)
    manifest = []
    counts = {"downloaded": 0, "exists": 0, "failed": 0}

    posts = data.get("posts", [])
    print(f"Loaded posts: {len(posts)}")

    for pi, post in enumerate(posts, 1):
        enriched = []
        for media in post.get("media", []):
            item = process_media(post, media, media_dir)
            enriched.append(item)
            status = item.get("download_status", "failed")
            counts[status] = counts.get(status, 0) + 1
            manifest.append({
                "post_id": str(post.get("post_id")),
                "created_at": post.get("created_at"),
                **item,
            })
            print(f"[{pi}/{len(posts)}] {post.get('post_id')} media#{media.get('index', 0)} -> {status}")
        post["media"] = enriched
        post["content_type"] = infer_content_type(post)
        post["needs_review"] = post["content_type"] == "UNKNOWN"

    data["schema_version"] = "selftrace-x-posts-v1.2-enriched"
    save_json(data, output_posts)
    save_json(manifest, output_manifest)

    print("\nDone")
    print(f"Posts: {len(posts)}")
    print(f"Media: {len(manifest)}")
    print(f"Downloaded: {counts.get('downloaded', 0)}")
    print(f"Already existed: {counts.get('exists', 0)}")
    print(f"Failed: {counts.get('failed', 0)}")
    print(f"Unknown posts: {sum(1 for p in posts if p.get('needs_review'))}")
    print(f"Enriched posts: {output_posts}")
    print(f"Manifest:       {output_manifest}")


if __name__ == "__main__":
    main()
