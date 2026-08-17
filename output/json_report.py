import json
from datetime import datetime
from pathlib import Path

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"


def build_match_report(
    home_team: str,
    away_team: str,
    league: str,
    kickoff: str,
    ml_probs: dict,
    bookmaker_probs: dict | None = None,
    polymarket_probs: dict | None = None,
    divergence_features: dict | None = None,
    claude_analysis: str | None = None,
    scorelines: dict | None = None,
) -> dict:
    """Assemble one match's prediction into a JSON-serializable dict."""
    probs = {k: round(float(v), 4) for k, v in ml_probs.items()}
    prediction = max(probs, key=probs.get)
    report = {
        "match": f"{home_team} vs {away_team}",
        "home_team": home_team,
        "away_team": away_team,
        "league": league,
        "kickoff": kickoff,
        "prediction": prediction,
        "confidence": probs[prediction],
        "probabilities": {
            "ml_model": probs,
            "bookmaker": bookmaker_probs,
            "polymarket": polymarket_probs,
        },
        "divergence": divergence_features,
        "scorelines": scorelines,
        "claude_analysis": claude_analysis,
    }
    return report


def write_json_report(matches: list[dict], report_date: str | None = None,
                      reports_dir: str | Path = REPORTS_DIR) -> Path:
    """
    Write the matchday prediction report to
    reports/predictions_<date>.json and return the path.
    """
    report_date = report_date or datetime.now().strftime("%Y-%m-%d")
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "report_date": report_date,
        "n_matches": len(matches),
        "matches": matches,
    }

    path = reports_dir / f"predictions_{report_date}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    return path
