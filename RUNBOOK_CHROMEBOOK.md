# Running the models on the Chromebook — daily runbook

Setup is already done (see SETUP_CHROMEBOOK.md). This is what you type
day to day. You never need to understand Linux for any of it.

## Every time you open the Terminal — start here
```
cd ~/soccer && source .venv/bin/activate
```
Your prompt must show `(.venv)` at the front. If it doesn't, the models
won't run — paste that line again.

## Terminal survival kit (first-timer notes)
- Paste = **Ctrl+Shift+V** (plain Ctrl+V does not work in the Terminal)
- Copy from the terminal = select text, then **Ctrl+Shift+C**
- `ls` is the letter L then S ("list files"); no command here uses digits
- Passwords never show while typing — it's recording, keep going
- Something stuck / scrolling forever? **Ctrl+C** stops it. Nothing breaks.
- Up-arrow recalls your last command — saves retyping the long loops
- Never `sudo rm` anything. You won't need to delete anything.

## THE ONE RULE
From your first scan on the Chromebook until you copy the database home,
**only the Chromebook touches stats.db.** The PC stays idle. Two copies
can't be merged. (Section 6 = how to bring it home.)

## 1. Settle — grade finished games (mornings, and after matchdays)
```
for lg in brazil argentina mls ligamx norway sweden korea china denmark czech scotland austria portugal netherlands belgium laliga championship epl seriea ligue1; do python value_scanner.py --league $lg --settle; done
```
~5 minutes. Each league prints `results ingested: N | spots graded: N`.
"0 graded" just means no bets on that league finished since last time.

## 2. Scan — fresh odds, fairs, new flags (before matchdays; before
##    early kickoffs to capture closes)
```
for lg in brazil argentina mls ligamx norway sweden korea china denmark czech scotland austria portugal netherlands belgium laliga championship epl seriea ligue1; do python value_scanner.py --league $lg; done
```
~10 minutes. Ends each league with `spots: N flagged this run; N total`.
A `[WARN] implausible edge skipped` line is fine — it's the 20% EV guard
catching a bad feed price, on purpose.

## 3. Rebuild the board
```
python build_board.py
```
Prints `payload rebuilt: N upcoming / N open bets / N settled`.

## 4. Publish to Lavish (via Claude Code)
Start Claude Code in the same folder:
```
claude
```
(first time only: it opens a browser to log in.) Then just say:
> settle and refresh the board

It knows the routine, the board URL, the memory notes — same as at home.
It will run steps 1-3 for you too, so on most days you can skip straight
to `claude` and ask. The commands above are for when you want to run
things yourself or if Claude Code is unavailable.

## 5. Czech anchors (the only manual data input)
Send/paste your Pinnacle screenshot to Claude Code and say "Czech" — it
edits `data/manual_anchors.json` and rescans. If you're doing it by hand,
the file is plain text: `nano data/manual_anchors.json`, edit,
Ctrl+O, Enter, Ctrl+X.

## 6. Coming home — bring the database back (do this once, at the end)
On the Chromebook: **Files app → Linux files → soccer → data → stats.db**
→ copy → paste into **OneDrive → WTA Model → football → data**, replacing
the old one. (Or upload it via onedrive.live.com if OneDrive isn't in the
Files app.) Once that finishes syncing, the PC is live again and the
Chromebook copy should not be used.

## If something goes wrong
| You see | It means | Do |
|---|---|---|
| `stats.db missing or suspiciously small` | DB isn't in `~/soccer/data/` | re-copy it (Setup step 2) |
| `KeyError: 'THE_ODDS_API_KEY'` / 401 | odds key missing from `.env` | `nano .env`, add the line |
| `command not found: python` | venv not active | `source .venv/bin/activate` |
| `[WARN] independent engine unavailable` | v2 skipped one league | harmless, ignore |
| a league prints `board: 0 matches` | nothing upcoming in the feed | normal midweek |
| Terminal froze | a scan is just slow | wait; or Ctrl+C and rerun |

Anything else: open `claude` and paste the error — it has full context.
