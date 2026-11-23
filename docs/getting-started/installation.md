# Installation

## Requirements

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/) package manager

## Install dependencies

```bash
git clone <repo-url>
cd dgm_saravanakumar

uv sync
```

To include dev tools (pytest, ruff, mypy):

```bash
uv sync --extra dev
```

## Install MkDocs (to view this documentation locally)

```bash
uv add --dev mkdocs-material
uv run mkdocs serve
```

Then open `http://127.0.0.1:8000` in your browser.
