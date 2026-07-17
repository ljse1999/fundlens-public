# Streamlit Community Cloud Deployment

## One-time setup

1. Push this repo to GitHub (`github.com/YOUR-USERNAME/fundlens-public`).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app**, select:
   - **Repository:** `YOUR-USERNAME/fundlens-public`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Click **Deploy**. Streamlit Cloud installs from `requirements.txt` and uses
   Python 3.12 from `.python-version`.

## Secrets (optional)

If you want the optional enrichment APIs (BEA, FMP, Tiingo, etc.) to work in
the live app:

1. Open the app in the Cloud dashboard.
2. Click **⋯ → Settings → Secrets**.
3. Paste the contents of `.streamlit/secrets.toml.example` with real values
   filled in.
4. Save. The app restarts and `config.py._load_env_chain` picks them up via
   `st.secrets`.

No secrets are required for the core analysis path.

## Resource limits (free tier)

- 1 GB RAM, shared CPU.
- Container sleeps after ~7 days idle (~30–60s cold start on next visit).
- Ephemeral filesystem — `reports/` writes vanish on container recycle
  (the in-session `st.download_button` path still works).
- "Re-run live" in the screen tab is capped at 50 funds per run.

## Updating the snapshot

After refreshing the source data:

```bash
python scripts/build_snapshot.py --funds <new-triaged-csv> --screen <new-checkpoint-csv> --out data
git add data/
git commit -m "Refresh IA snapshot"
git push
```

Streamlit Cloud auto-redeploys on push to `main`.
