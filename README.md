# FundLens

> Factor decomposition and alpha analysis for UK/European equity mutual funds.

**Live demo:** [https://YOUR-APP-NAME.streamlit.app](https://YOUR-APP-NAME.streamlit.app)

FundLens decomposes a fund's returns against Fama-French factor models, tests
whether any alpha is statistically genuine, and surfaces rule-based diligence
flags — all from a fund's ISIN.

## What it does

- **Analyse a fund:** enter an ISIN (or search by name), pick a benchmark and
  factor region, and get a full triage report: factor decomposition, alpha
  significance (with bootstrap), holdings analytics, and diligence flags.
- **Screen funds:** rank a universe of UK/European equity funds by alpha
  significance. The default view loads from a committed snapshot (instant);
  a "Re-run live" button fetches fresh data.

## Try it

Open the [live demo](https://YOUR-APP-NAME.streamlit.app). No signup required.

For the **Analyse** tab, paste a fund ISIN such as:

- `GB00B41YBW71` — Fundsmith Equity
- `IE00BJSPMJ28` — Lindsell Train Global Equity
- `GB00B3X7QG63` — Vanguard FTSE UK All Share

For the **Screen** tab, the "Snapshot" source shows ~1,450 IA-listed equity
funds with pre-computed alpha metrics.

## Self-hosting

```bash
git clone https://github.com/ljse1999/fundlens-public.git
cd fundlens-public
pip install -r requirements.txt
streamlit run app.py
```

Python 3.12+ required. No API keys are needed for the core analysis path
(Yahoo Finance, Ken French, and FRED-basic all work keyless).

### Optional API keys (enrichment only)

Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and fill in
values. See the file for the full list.

## How it works

See [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) for the factor model, alpha
testing procedure, and flag definitions.

## Deployment

See [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) for the Streamlit Community
Cloud setup.

## Data

The screen tab's snapshot lives in [`data/`](data/) and is described in
[`data/README.md`](data/README.md). Public fund names, management companies,
ISINs, and derived alpha metrics only.

## Disclaimer

This tool is for research and educational purposes only. It is **not
investment advice**. FundLens analyses public fund data using published
methodology; past alpha is not indicative of future performance.

## License

MIT — see [`LICENSE`](LICENSE).
