# Streamlit Cloud Deployment

The dashboard is deployed and live at:
**[https://l85nqzw6mxzpf2gwgxcb9r.streamlit.app/](https://l85nqzw6mxzpf2gwgxcb9r.streamlit.app/)**

---

## Architecture

```
GitHub repo (code)          Hugging Face Hub (weights)
       │                            │
       ▼                            ▼
Streamlit Cloud  ──── downloads at inference time ────▶ runs inference
```

- **Code** lives in GitHub and is auto-deployed to Streamlit Cloud on every push.
- **Model weights** (`.pt` files) are hosted on Hugging Face Hub at `sarav95/ecg-anomaly-weights` and downloaded at runtime on first inference.

---

## Step-by-Step Deployment Guide

### Step 1 — Push code to GitHub

Push the full project to a GitHub repository. The app entry point is `app.py` at the repo root.

!!! warning "Remove uv.lock"
    Delete `uv.lock` from the repo before pushing (`git rm uv.lock`). Streamlit Cloud detects multiple requirements files and may use `uv.lock` instead of `requirements.txt`, causing missing packages.

### Step 2 — Upload model weights to Hugging Face Hub

Model weights are too large to push to GitHub. Create a model repository at `sarav95/ecg-anomaly-weights` on Hugging Face and upload the `.pt` files organized into two subfolders — `ptbxl/` for PTB-XL weights and `mimic/` for MIMIC-IV-ECG weights.

The inference code automatically checks for local weights first. If not found (as on Streamlit Cloud), it downloads them from the HF Hub repo at runtime and caches them for subsequent runs.

### Step 3 — Deploy on Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. Click **New app**.
3. Select the repository, branch (`master`), and set the main file path to `app.py`.
4. Click **Deploy**.

Streamlit Cloud automatically redeploys on every `git push` to the connected branch. Watch live build logs via **Manage app → Logs** at the bottom right of the app.

### Step 4 — Set the HF token secret (if model repo is private)

Go to app → top-right `⋮` → **Settings** → **Secrets** and add:

```toml
HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxx"
```

The app reads this token and passes it to the HF Hub download call when fetching weights.

---

## Auto-deploy on push

Every `git push` to the connected branch triggers a Streamlit Cloud rebuild automatically — no manual action needed. The first deploy takes longer (~5–10 minutes) due to downloading PyTorch and CUDA libraries.
