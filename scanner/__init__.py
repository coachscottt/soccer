"""Fair-value scanner package.

Split from the original monolithic value_scanner.py (2026-07-25):
  asian_totals.py - payoff-exact Asian totals math (lines .0/.25/.5/.75)
  calibration.py  - Dixon-Coles matrix calibrated to sharp anchors
  database.py     - schema + connection for the scanner tables
  scanner.py      - odds feeds, anchors, flagging, boards (scan mode)
  settlement.py   - results, grading, CLV, ledger reports (settle mode)

CLI entry point stays `python value_scanner.py` (thin shim).
"""
from .scanner import scan
from .settlement import settle, void_events

__all__ = ["scan", "settle", "void_events"]
