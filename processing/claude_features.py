import anthropic
import json
from pathlib import Path
from dotenv import load_dotenv

# Load football/.env first, then fall back to any .env up the tree
load_dotenv(Path(__file__).resolve().parents[1] / ".env")
load_dotenv()

_client = None


def _get_client() -> anthropic.Anthropic:
    # Lazy init so importing this module doesn't require ANTHROPIC_API_KEY
    global _client
    if _client is None:
        _client = anthropic.Anthropic()  # key from ANTHROPIC_API_KEY env variable
    return _client


MODEL = "claude-sonnet-5"  # article used claude-sonnet-4-20250514 (deprecated)

# Structured-output schema: the API guarantees the response is valid JSON
# matching this shape, so no scrape-the-braces fallback parsing is needed.
_SCORE = {"type": "number"}
MATCHUP_SCHEMA = {
    "type": "object",
    "properties": {
        "home_attack_strength": _SCORE,
        "home_defense_strength": _SCORE,
        "away_attack_strength": _SCORE,
        "away_defense_strength": _SCORE,
        "home_momentum": _SCORE,
        "away_momentum": _SCORE,
        "match_intensity_prediction": _SCORE,
        "upset_probability": _SCORE,
        "home_win_confidence": _SCORE,
        "draw_likelihood": _SCORE,
        "reasoning": {"type": "string"},
    },
    "required": [
        "home_attack_strength", "home_defense_strength",
        "away_attack_strength", "away_defense_strength",
        "home_momentum", "away_momentum",
        "match_intensity_prediction", "upset_probability",
        "home_win_confidence", "draw_likelihood", "reasoning",
    ],
    "additionalProperties": False,
}


def _fmt(value) -> str:
    """Format a stat as 2-decimal string; pass 'N/A' through for missing values."""
    return f"{value:.2f}" if isinstance(value, (int, float)) else "N/A"


def claude_analyze_matchup(
    home_team: str,
    away_team: str,
    home_form: dict,
    away_form: dict,
    league: str,
) -> dict:
    """
    Ask Claude to evaluate contextual match factors
    that are difficult to extract from numerical data.

    Returns JSON with scores on a 0.0-1.0 scale.
    """
    prompt = f"""You are an expert football match analyst. Analyze the upcoming match
and return ONLY JSON (no markdown, no comments) with the following scores
on a scale from 0.0 to 1.0:

Match: {home_team} (home) vs {away_team} (away)
League: {league}

{home_team} stats over last 5 matches:
- Avg goals scored: {_fmt(home_form.get('avg_GF'))}
- Avg goals conceded: {_fmt(home_form.get('avg_GA'))}
- Avg shots: {_fmt(home_form.get('avg_Shots'))}
- Avg shots on target: {_fmt(home_form.get('avg_SoT'))}
- Form (avg points): {_fmt(home_form.get('Form'))}

{away_team} stats over last 5 matches:
- Avg goals scored: {_fmt(away_form.get('avg_GF'))}
- Avg goals conceded: {_fmt(away_form.get('avg_GA'))}
- Avg shots: {_fmt(away_form.get('avg_Shots'))}
- Avg shots on target: {_fmt(away_form.get('avg_SoT'))}
- Form (avg points): {_fmt(away_form.get('Form'))}

Return JSON strictly in the format:
{{
    "home_attack_strength": <float>,
    "home_defense_strength": <float>,
    "away_attack_strength": <float>,
    "away_defense_strength": <float>,
    "home_momentum": <float>,
    "away_momentum": <float>,
    "match_intensity_prediction": <float>,
    "upset_probability": <float>,
    "home_win_confidence": <float>,
    "draw_likelihood": <float>,
    "reasoning": "<brief 1-2 sentence explanation>"
}}"""

    message = _get_client().messages.create(
        model=MODEL,
        max_tokens=1024,
        thinking={"type": "disabled"},
        output_config={
            "effort": "low",
            "format": {"type": "json_schema", "schema": MATCHUP_SCHEMA},
        },
        messages=[{"role": "user", "content": prompt}],
    )

    if message.stop_reason == "refusal":
        return {}

    # With output_config.format the first block is guaranteed valid JSON
    response_text = next(
        (b.text for b in message.content if b.type == "text"), ""
    ).strip()
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        return {}
