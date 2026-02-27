from pydantic import BaseModel, Field
from typing import Literal, Optional, List, Dict, Any

Decision = Literal["bid", "pass"]

class AgentProfile(BaseModel):
    name: str
    budget_eur: int
    preferences: str  # free text: neighborhoods, rooms, must-haves
    strategy: Literal["aggressive", "conservative", "analytical"]

class BidResponse(BaseModel):
    agent_name: str
    decision: Decision
    bid_eur: Optional[int] = None
    rationale: str = Field(..., description="Short explanation grounded in property context")
    max_willing_eur: Optional[int] = None  # useful for debugging/analysis

class AuctionResult(BaseModel):
    property_id: str
    winner: Optional[str] = None
    final_price_eur: Optional[int] = None
    history: List[Dict[str, Any]]