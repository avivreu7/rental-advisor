def explain_simple(inputs: dict, pred_price: float, ref_stats: dict) -> str:
    """
    Lightweight, non-LLM explanation in simple English.
    Summarizes the predicted price and adds quick context from local stats and features.
    """
    nb = inputs.get("neighbourhood")
    rt = inputs.get("room_type")
    b = inputs.get("bedrooms")
    ba = inputs.get("bathrooms")

    parts = [f"Predicted nightly price: ~${pred_price:,.0f}."]
    if ref_stats:
        parts.append(
            f"In {nb} · {rt}, the median is ~${ref_stats.get('median', 0):,.0f} "
            f"(IQR ${ref_stats.get('p25', 0):,.0f}–${ref_stats.get('p75', 0):,.0f})."
        )

    bullets = []
    if rt and "Entire home" in rt:
        bullets.append("Entire homes typically price higher than private/shared rooms.")
    if b and b >= 2:
        bullets.append("More bedrooms generally support a higher price.")
    if ba and ba >= 2:
        bullets.append("More bathrooms increase attractiveness for families/groups.")

    if bullets:
        parts.append("Key factors: " + " ".join(bullets))

    return " ".join(parts)
