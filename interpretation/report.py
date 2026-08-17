from processing.claude_features import _get_client, MODEL


def generate_prediction_report(
    home_team: str,
    away_team: str,
    model_proba: dict,
    stats: dict,
    league: str,
) -> str:
    """
    Generate a detailed analytical report
    using Claude based on model probabilities
    and team statistics.
    """
    prompt = f"""You are a professional football analyst. Based on the machine
learning model data and team statistics, write a concise but
insightful analytical report on the upcoming match.

## Model Data

Match: **{home_team}** vs **{away_team}** ({league})

Model probabilities (ML Ensemble):
- {home_team} win: {model_proba['home_win']:.1%}
- Draw: {model_proba['draw']:.1%}
- {away_team} win: {model_proba['away_win']:.1%}

{home_team} stats (last 5 matches):
- Goals scored (avg): {stats['home_avg_GF']:.2f}
- Goals conceded (avg): {stats['home_avg_GA']:.2f}
- Shots on target (avg): {stats['home_avg_SoT']:.1f}
- Form (avg points): {stats['home_Form']:.2f}

{away_team} stats (last 5 matches):
- Goals scored (avg): {stats['away_avg_GF']:.2f}
- Goals conceded (avg): {stats['away_avg_GA']:.2f}
- Shots on target (avg): {stats['away_avg_SoT']:.1f}
- Form (avg points): {stats['away_Form']:.2f}

## Task

Write an analytical report that includes:
1. Key factors affecting the prediction
2. Strengths and weaknesses of each team
3. Most likely outcome prediction
4. Confidence level (high / medium / low)
5. Potential risks and upset scenarios

Write concisely, professionally, no filler."""

    message = _get_client().messages.create(
        model=MODEL,
        max_tokens=1000,
        thinking={"type": "disabled"},
        messages=[{"role": "user", "content": prompt}],
    )

    if message.stop_reason == "refusal":
        return ""

    return next(
        (b.text for b in message.content if b.type == "text"), ""
    )
