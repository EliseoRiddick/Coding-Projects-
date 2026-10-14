# Prompt for your Claude Project (upload `bundles/*.md` as project knowledge first)

You are helping me turn a large set of video transcripts into a structured trading playbook of about 30 pages (~15,000 words). Work in three passes and wait for me to say "continue" between passes.

**Pass 1 – Extraction.** For each bundle file, list every distinct concept, rule, model, and term the speaker teaches (market structure, liquidity, fair value gaps, order blocks, premium/discount, time-of-day and day-of-week rules, entry models, risk rules). Note which video(s) it came from. Output a compact table per bundle.

**Pass 2 – Consolidation.** Merge the tables. Resolve duplicates and contradictions (say which version appears more often or later). Flag anything that is vague, unstated, or that you are inferring rather than reading directly, marking it `[LOW CONFIDENCE]`.

**Pass 3 – Playbook.** Write the playbook in your own words (no long verbatim quotes) with these chapters:
1. Philosophy and how the market is framed
2. Market structure (swings, shifts, breaks)
3. Liquidity (equal highs/lows, old highs/lows, stop runs)
4. Price delivery arrays (fair value gaps, order blocks, breakers, premium/discount, optimal trade entry)
5. Time: sessions and killzones (times in New York), daily bias, weekly range profiles, day-of-week tendencies
6. Multi-timeframe workflow (what to read on higher timeframes, what to execute on lower)
7. Entry models, step-by-step, each with an explicit checklist
8. Stops, targets, and risk management
9. Pre-trade and post-trade checklists
10. Glossary

Make every rule testable: state inputs (levels, times, timeframes), the condition, and the expected action. Number the rules (R1, R2, ...) so a trading agent can cite them. End with a one-page "rules index".
