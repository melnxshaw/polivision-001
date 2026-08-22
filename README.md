# NBP Vaults — Life Insurance Intelligence (Saffron-styled v3)

A third, completely separate app — same real data and features as your other versions,
with a new UI inspired by saffron-griflan.netlify.app: warm amber/saffron accent on a
dark background, bold serif hero headline, stat cards, and an FAQ-accordion style
documentation tab.

**Note on the visual reference:** the UI here was built from the site's visible content
and structure (bold hero statement, stat counters, data tables, FAQ accordion) plus the
warm amber "saffron" color implied by the brand name/token — not a pixel-exact copy,
since exact colors/fonts aren't extractable from fetched page text alone.

## Files
```
app.py               # Streamlit app (entry point) — Saffron-styled UI
data_pipeline.py       # Data ingestion & cleaning (same as other versions)
forecasting.py          # Models, metrics, backtesting (same as other versions)
combined_clean.csv      # Built-in real dataset
requirements.txt
runtime.txt
```

## Deploy as a brand-new, third app
1. Create a NEW GitHub repository, e.g. `nbp-vaults-saffron`
2. Upload all 6 files via GitHub's web uploader
3. Go to https://share.streamlit.io → **New app** → this new repo → main file `app.py` → **Deploy**
4. You'll get a third, independent public URL — your other two apps are untouched

## Run locally to preview first (optional)
```bash
pip install -r requirements.txt
streamlit run app.py
```
