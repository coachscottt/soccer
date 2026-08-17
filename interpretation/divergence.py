from processing.claude_features import _get_client, MODEL


def claude_analyze_divergence(
    match: str,
    bookmaker: dict,
    polymarket: dict,
    ml_model: dict,
    poly_liquidity: float,
    poly_volume_24h: float,
) -> str:
    """
    Claude analyzes divergences between three sources
    and proposes an interpretation.
    """
    prompt = f"""You are a senior sports analyst. You have three probability
sources for a football match. Analyze the divergences.

**Match:** {match}

| Source | Home | Draw | Away |
|---|---|---|---|
| Bookmaker (Bet365) | {bookmaker['home']:.1%} | {bookmaker['draw']:.1%} | {bookmaker['away']:.1%} |
| Polymarket | {polymarket['home']:.1%} | {polymarket['draw']:.1%} | {polymarket['away']:.1%} |
| ML Model | {ml_model['home']:.1%} | {ml_model['draw']:.1%} | {ml_model['away']:.1%} |

**Polymarket metadata:**
- Liquidity: ${poly_liquidity:,.0f}
- 24h volume: ${poly_volume_24h:,.0f}

**Task:**
1. Where are the main divergences and what might they mean?
2. Which source should be trusted more in this case and why?
3. Are there signs of insider activity on Polymarket?
   (unusual volume, sharp probability shift)
4. What final prediction would you give and with what confidence?

Be specific, no filler. 5-8 sentences."""

    message = _get_client().messages.create(
        model=MODEL,
        max_tokens=1000,  # article's 600 truncates the 4-part answer mid-sentence
        thinking={"type": "disabled"},
        messages=[{"role": "user", "content": prompt}],
    )

    if message.stop_reason == "refusal":
        return ""

    return next(
        (b.text for b in message.content if b.type == "text"), ""
    )
