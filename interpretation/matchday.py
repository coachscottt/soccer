from processing.claude_features import _get_client, MODEL


def analyze_matchday(matches: list[dict]) -> str:
    """
    Analyze an entire matchday with a single Claude call.
    More efficient than separate requests for each match.

    (Article annotated the return as list[dict], but the function
    returns Claude's formatted report text.)
    """
    matches_text = ""
    for i, m in enumerate(matches, 1):
        matches_text += f"""
{i}. {m['home']} vs {m['away']}
   ML prediction: H={m['prob_H']:.0%} | D={m['prob_D']:.0%} | A={m['prob_A']:.0%}
   Home form: {m['home_form']:.2f} | Away form: {m['away_form']:.2f}
"""

    prompt = f"""Analyze the upcoming matchday. For each match, provide:
- Prediction (1X2)
- Confidence (⭐ low, ⭐⭐ medium, ⭐⭐⭐ high)
- Brief comment (1 sentence)

Matches:
{matches_text}

Return in table format. At the end, add the 1-2 best picks of the matchday (highest confidence)."""

    message = _get_client().messages.create(
        model=MODEL,
        max_tokens=1500,
        thinking={"type": "disabled"},
        messages=[{"role": "user", "content": prompt}],
    )

    if message.stop_reason == "refusal":
        return ""

    return next(
        (b.text for b in message.content if b.type == "text"), ""
    )
