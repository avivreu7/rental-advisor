from typing import Dict, Any, Optional
from openai import OpenAI

SYSTEM_PROMPT = (
    "You are a helpful real-estate pricing explainer for San Francisco Airbnb listings. "
    "Explain in 2–4 concise sentences, in clear, simple English, why a predicted price is reasonable. "
    "Use the given features (neighbourhood, room type, bedrooms, bathrooms) and local stats (median, IQR) when available. "
    "Avoid overconfident or absolute claims; keep it practical and user-friendly."
)

def build_client(api_key: str) -> Optional[OpenAI]:
    return OpenAI(api_key=api_key) if api_key else None

def explain_llm(
    client: Optional[OpenAI],
    model: str,
    inputs: Dict[str, Any],
    pred_price: float,
    ref_stats: Dict[str, Any]
) -> str:
    if not client or not model:
        return ""

    payload = {
        "inputs": inputs,
        "predicted_price": float(pred_price),
        "stats": ref_stats or {}
    }

    try:
        resp = client.chat.completions.create(
            model=model,  # e.g., "gpt-5-nano"
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": str(payload)}
            ]
            # IMPORTANT: no temperature here for gpt-5-nano
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        return ""
