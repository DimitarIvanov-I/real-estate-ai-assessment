import json
from openai import OpenAI
from pydantic import ValidationError
from src.agents.schemas import AgentProfile, BidResponse

class BuyerAgent:
    def __init__(self, profile: AgentProfile, client: OpenAI, model: str):
        self.profile = profile
        self.client = client
        self.model = model

    def decide_bid(self, property_context: dict, current_price_eur: int, min_increment_eur: int) -> BidResponse:
        """
        property_context: dict with property_id + chunks + metadata
        """
        ml_pred = property_context.get("ml", {}).get("predicted_price_eur")

        prompt = f"""
You are a real-estate buyer agent in an auction.

Agent profile:
Name: {self.profile.name}
Budget (EUR): {self.profile.budget_eur}
Strategy: {self.profile.strategy}
Preferences: {self.profile.preferences}

Auction state:
Property: {property_context.get("property_id")}
Current highest bid (EUR): {current_price_eur}
Minimum increment (EUR): {min_increment_eur}

ML signal (fair value estimate):
predicted_price_eur: {ml_pred}

Rules:
- If predicted_price_eur is provided, treat it as an estimate of fair market value.
- Set max_willing_eur based on: min(budget, predicted_price_eur adjusted for risk/strategy).
  Examples:
  - aggressive: up to ~100% of predicted_price_eur
  - conservative: up to ~95-98% of predicted_price_eur
  - legal risks (Act 15, pending inheritance): reduce max_willing_eur further
- If bidding, bid must be >= current_price_eur + min_increment_eur
- Never exceed your budget or your max_willing_eur
{{
  "agent_name": "{self.profile.name}",
  "decision": "bid" or "pass",
  "bid_eur": integer or null,
  "rationale": string,
  "max_willing_eur": integer or null
}}

Be concise. Ground your rationale in the provided property info.
""".strip()

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Return ONLY valid JSON. No markdown."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )

        raw = resp.choices[0].message.content.strip()

        try:
            data = json.loads(raw)
            bid = BidResponse(**data)
        except (json.JSONDecodeError, ValidationError) as e:
            # safe fallback: pass if output is malformed
            return BidResponse(
                agent_name=self.profile.name,
                decision="pass",
                bid_eur=None,
                rationale=f"Model output invalid; passing. Error: {type(e).__name__}",
                max_willing_eur=None,
            )

        # hard validation
        if bid.decision == "bid":
            if bid.bid_eur is None:
                return BidResponse(agent_name=self.profile.name, decision="pass", rationale="No bid_eur provided.", max_willing_eur=bid.max_willing_eur)
            if bid.bid_eur > self.profile.budget_eur:
                return BidResponse(agent_name=self.profile.name, decision="pass", rationale="Bid would exceed budget.", max_willing_eur=bid.max_willing_eur)
            if bid.bid_eur < current_price_eur + min_increment_eur:
                return BidResponse(agent_name=self.profile.name, decision="pass", rationale="Bid below min increment.", max_willing_eur=bid.max_willing_eur)
            if bid.max_willing_eur is not None and bid.bid_eur > bid.max_willing_eur:
                return BidResponse(agent_name=self.profile.name, decision="pass", bid_eur=None, rationale="Bid would exceed max_willing_eur.", max_willing_eur=bid.max_willing_eur)

        return bid