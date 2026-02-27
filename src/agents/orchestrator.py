from typing import List, Dict, Any, Optional
from src.agents.schemas import AuctionResult
import random


class AuctionOrchestrator:
    def __init__(self, agents, min_increment_eur: int, max_rounds: int, shuffle_each_round: bool = True):
        self.agents = agents
        self.min_increment_eur = min_increment_eur
        self.max_rounds = max_rounds
        self.shuffle_each_round = shuffle_each_round

    def run_auction(self, property_context: Dict[str, Any], starting_bid_eur: int) -> AuctionResult:
        current_price = starting_bid_eur
        current_winner: Optional[str] = None
        history: List[Dict[str, Any]] = []

        for rnd in range(1, self.max_rounds + 1):
            any_new_bid = False

            agents = list(self.agents)
            if self.shuffle_each_round:
                random.shuffle(agents)

            for agent in agents:
                bid = agent.decide_bid(property_context, current_price, self.min_increment_eur)

                history.append({
                    "round": rnd,
                    "agent": bid.agent_name,
                    "decision": bid.decision,
                    "bid_eur": bid.bid_eur,
                    "current_price_before": current_price,
                    "rationale": bid.rationale,
                    "max_willing_eur": bid.max_willing_eur,
                })

                # accept bid only if it raises price AND is not from current winner
                if (
                        bid.decision == "bid"
                        and bid.bid_eur is not None
                        and bid.bid_eur > current_price
                        and bid.agent_name != current_winner
                ):
                    current_price = bid.bid_eur
                    current_winner = bid.agent_name
                    any_new_bid = True

            if not any_new_bid:
                break

        return AuctionResult(
            property_id=property_context.get("property_id", "UNKNOWN"),
            winner=current_winner,
            final_price_eur=current_price if current_winner else None,
            history=history,
        )