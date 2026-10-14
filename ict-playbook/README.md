# ICT transcripts → playbook → trade-idea agent

Three stages. Stage 1 runs on **your computer** (YouTube often blocks cloud IPs).

```bash
cd ict-playbook
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 1. Pull captions
```bash
python captions/fetch_captions.py --out ./transcripts --limit 3   # test on 3 videos
python captions/fetch_captions.py --out ./transcripts             # whole channel (resumable)
```
Saves `transcripts/<date>_<id>_<title>.txt`. Manual captions are preferred, auto-captions are the fallback;
videos with neither are listed in `transcripts/failed.log`. Re-run any time to pick up new videos.
For personal study; respect YouTube's terms and the creator's rights.

## 2. Build the playbook (in a Claude Project)
```bash
python captions/bundle_transcripts.py --src ./transcripts --out ./bundles
```
Upload `bundles/*.md` to a Claude Project as knowledge, then paste `captions/playbook_prompt.md`.
It runs three passes (extract → consolidate → write ~30 pages with numbered rules R1, R2, …).
Save the result as `playbook.md`.

## 3a. Colab notebook (easiest way to use the agent)
Open `colab/ict_analyst.ipynb` in Google Colab (upload it, or File > Open notebook > GitHub). It:
- reads your playbook straight from the Google Drive doc (edit the doc, re-run, and the new version is used),
- takes OHLC CSVs of any file name (the timeframe is detected from bar spacing),
- optionally takes up to 8 chart screenshots and pasted economic data,
- computes levels/sessions/weekday context in Python, then asks Claude for ideas and shows them formatted.
Add your key as a Colab secret named `ANTHROPIC_API_KEY`. Tick DRY_RUN to preview the request for free.

## 3b. Trade-idea agent from the command line
Put one CSV per timeframe in a folder, named by timeframe (`1m.csv`, `15m.csv`, `1h.csv`, `4h.csv`, `1d.csv`).
Columns: a time column (`timestamp`/`datetime`/`time`/`date`, naive local, tz-aware, or unix epoch) plus
`open,high,low,close`. Missing higher timeframes can be derived with `--derive 1h,4h,1d`.

```bash
export ANTHROPIC_API_KEY=...
cd agent
python -m ict_agent.cli --data-dir ../data --playbook ../playbook.md --symbol NQ --derive 1h,4h,1d --dry-run
python -m ict_agent.cli --data-dir ../data --playbook ../playbook.md --symbol NQ --derive 1h,4h,1d \
    --images chart1.png chart2.png --econ calendar.txt
# replay a past moment without lookahead:
python -m ict_agent.cli ... --asof "2025-03-12 08:45"
```
Python computes levels (previous day/week/month high-low, opens, swings, equal highs/lows, unfilled FVGs,
premium/discount), session, day of week and week profile; Claude only reasons over that snapshot using your
playbook (cached in the system prompt). Output is JSON saved to `ideas/`. Session windows live in
`agent/ict_agent/sessions.py` - edit them to match your playbook.

Tests: `python -m pytest agent/tests`. Educational analysis only, not financial advice.
