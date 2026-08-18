"""Fair-value scanner CLI (thin shim).

The implementation lives in the scanner/ package:
  scanner/scanner.py      scan mode (feeds, anchors, boards, flagging)
  scanner/settlement.py   settle mode (results, grading, CLV, ledger)
  scanner/calibration.py  market-calibrated Dixon-Coles matrix
  scanner/asian_totals.py payoff-exact Asian totals math
  scanner/database.py     schema for the stats.db scanner tables

Run:  python value_scanner.py --league <key> [--min-ev 0.03]
      python value_scanner.py --league <key> --settle
      python value_scanner.py --void <event_id> ...   # postponed match
League keys come from mls_report.LEAGUES.
"""
import argparse

from mls_report import LEAGUES
from scanner import scan, settle, void_events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", choices=list(LEAGUES), default="brazil")
    ap.add_argument("--min-ev", type=float, default=0.03)
    ap.add_argument("--settle", action="store_true")
    ap.add_argument("--void", nargs="+", metavar="EVENT_ID",
                    help="mark every ungraded quote on these events void "
                         "(postponed/cancelled match: the bet never stood)")
    args = ap.parse_args()
    cfg = LEAGUES[args.league]
    if args.void:
        void_events(args.void)
    elif args.settle:
        settle(args, cfg)
    else:
        scan(args, cfg)


if __name__ == "__main__":
    main()
