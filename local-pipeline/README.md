# SelfTrace Local Pipeline

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Install Ollama separately, then:
```bash
ollama pull qwen3-vl:4b
```

## 1. Put exported JSON here
`data/raw/x_posts.json`

## 2. Download images
```bash
python scripts/download_media.py \
  --input data/raw/x_posts.json \
  --media-dir data/media \
  --output-posts data/processed/posts_enriched.json \
  --output-manifest data/processed/media_manifest.json
```

## 3. Analyze one multimodal post
```bash
python scripts/analyze_post.py \
  --input data/processed/posts_enriched.json \
  --post-id 2092794113787646099 \
  --model qwen3-vl:4b
```
