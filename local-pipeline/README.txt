SelfTrace path fix v1.4

Replace these two files in your local-pipeline/scripts/ directory:
- scripts/download_media.py
- scripts/analyze_post.py

Key change:
PROJECT_ROOT = Path(__file__).resolve().parents[1]

All relative paths such as data/analysis/... and data/media/... are resolved against local-pipeline/, not the terminal's current working directory.

analyze_post.py now defaults to qwen3-vl:8b.

You can run from anywhere, e.g.:
python /path/to/local-pipeline/scripts/analyze_post.py --post-id 2092794113787646099

The output will always go to:
/path/to/local-pipeline/data/analysis/2092794113787646099.json
