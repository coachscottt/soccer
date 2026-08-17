# Running the scanner on a Chromebook (Linux mode)

One-time setup ~15 min. Everything below is typed into the Chromebook's
**Terminal** app (the Linux one, appears after enabling Linux).

## 0. Enable Linux (once)
Settings → About ChromeOS → Developers → Linux development environment →
Turn on. Accept defaults. Open the **Terminal** app when it finishes.

## 1. Get the code
```bash
sudo apt update && sudo apt install -y git python3 python3-venv python3-pip
git clone https://github.com/coachscottt/soccer.git
cd soccer
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt        # ~2-3 min
```

## 2. Bring the two things NOT in the repo (secrets + database)
On the Chromebook, open the **Files** app → OneDrive (or Google Drive if
you copied there) → navigate to `WTA Model/football/`.
- Right-click **`data/stats.db`** (~265MB) → Copy → paste into
  **Linux files → soccer → data/**  (drag-and-drop into "Linux files" works)
- Do the same with **`.env`** → into **Linux files → soccer/**
  (Files app hides dot-files: menu ⋮ → "Show hidden files")

Then add the odds key to that .env — it lived in the PARENT folder's .env
on the PC. Open `WTA Model/.env` in OneDrive, copy the
`THE_ODDS_API_KEY=...` line, and append it:
```bash
nano .env        # paste the THE_ODDS_API_KEY line at the bottom, Ctrl+O, Enter, Ctrl+X
```

## 3. Smoke test (must all pass before you leave home)
```bash
source .venv/bin/activate      # (every new terminal)
python -c "from scanner.database import get_conn; get_conn().close(); print('DB OK')"
python value_scanner.py --league korea            # small league, ~1 min
python value_scanner.py --league korea --settle
python build_board.py
```
If the first line prints `DB OK`, the copy worked. If it says "missing or
suspiciously small", the DB didn't land in `soccer/data/` — fix that first.

## 4. Daily rhythm on the road (identical to home)
```bash
cd ~/soccer && source .venv/bin/activate
for lg in brazil argentina mls ligamx norway sweden korea china denmark czech \
          scotland austria portugal netherlands belgium laliga championship \
          epl seriea ligue1; do python value_scanner.py --league $lg --settle; done
for lg in ...same list...; do python value_scanner.py --league $lg; done
python build_board.py
```
Then publish `.lavish/scanner.html` to the Lavish board from a Claude Code
session (`claude` in the same folder; install once with
`curl -fsSL https://claude.ai/install.sh | bash`).

## 5. THE ONE RULE — the database has exactly one home
Once you scan on the Chromebook, **the Chromebook copy of `stats.db` is
the live one for the whole trip.** Do not run anything on the PC.
When you're back: copy `soccer/data/stats.db` from Linux files back to
OneDrive `WTA Model/football/data/stats.db` (overwrite). Then the PC is
live again. Two diverging copies cannot be merged.

Czech anchors: still send Pinnacle screenshots; edit
`data/manual_anchors.json` the same way. Everything else needs no manual input.
