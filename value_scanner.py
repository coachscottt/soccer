"""Fair-value scanner CLI (thin shim).

The implementation lives in the scanner/ package:
  scanner/scanner.py      scan mode (feeds, anchors, boards, flagging)
  scanner/settlement.py   settle mode (results, grading, CLV, ledger)
  scanner/calibration.py  market-calibrated Dixon-Coles matrix
  scanner/asian_totals.py payoff-exact Asian totals math
  scanner/database.py     schema for the stats.db scanner tables

Run:  python value_scanner.py --league <key> [--min-ev 0.03]
      python value_scanner.py --league <key> --settle
League keys come from mls_report.LEAGUES.
"""
import argparse

from mls_report import LEAGUES
from scanner import scan, settle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", choices=list(LEAGUES), default="brazil")
    ap.add_argument("--min-ev", type=float, default=0.03)
    ap.add_argument("--settle", action="store_true")
    args = ap.parse_args()
    cfg = LEAGUES[args.league]
    if args.settle:
        settle(args, cfg)
    else:
        scan(args, cfg)


if __name__ == "__main__":
    main()
