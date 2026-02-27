******Architecture Document — Real Estate Agents Tech Assessment******
**Overview**

This project implements a small real-estate property search and auction simulation system driven by:
1. Unstructured property documents
2. RAG retrieval (embeddings + vector database) for search/chat
3. Multi-agent auction (LLM buyer agents with different strategies)
4. ML price prediction (Option A) to estimate fair value and influence bidding

All functionality is accessible via CLI scripts.

**High-Level Data Flow**

1. Generate / store property docs
- Input: `data/raw_properties/property_XXX.md`
- Each file contains a short “header” section with structured fields (price, rooms, size, neighborhood, etc.) plus free text description.

2. Indexing (RAG)
- Script: src/m2_index.py
- Loads documents from RAW_PROPERTIES_DIR
- Extracts structured metadata from header lines using regex mapping: 
  - property_id, neighborhood, rooms, size_sqm, price_eur, property_type, etc.
- Splits each document into chunks using RecursiveCharacterTextSplitter
- Embeds chunks with OpenAI embeddings and persists them to Chroma:
  - Persistent directory: `CHROMA_DIR`
  - Collection: `COLLECTION_NAME`

3. Retrieval / Chat (RAG)

Script: `src/m2_retrieve.py` (debug retrieval)
Script: `src/m3_chat_cli.py` (interactive chat)

Given a query, the system:
- runs vector similarity search in Chroma
- returns top-matching chunks + metadata
- optionally uses a lightweight running preferences summary to handle follow-ups
The LLM answers using only retrieved chunks (grounded responses).

4. Auction Simulation (Multi-agent)

Script: `src/agents/simulate_auction.py`

For each auctioned property:
- fetches property context from Chroma (chunks + metadata)
- computes a starting bid from asking price
- runs an auction loop controlled by AuctionOrchestrator
- logs the full round-by-round decision history

5. ML Bonus (Option A) — Price Prediction

* Dataset builder: `src/ml/build_dataset_from_chroma.py`
  - Pulls metadata from Chroma and deduplicates by `property_id`
  - Produces tabular dataset: `data/ml/property_dataset.csv`
* Model trainer: src/ml/train_price_model.py
  - Trains a regression model predicting price_eur from:
    - numeric: rooms, size_sqm
    - categorical: neighborhood, property_type
  - Uses cross-validation due to small dataset size
- Saves model: `models/price_model.joblib`
* Predictor: `src/ml/predict_price.py`
  - Loads the saved model
  - Predicts fair value per property from auction metadata

* Auction integration:
  - `simulate_auction.py` injects `predicted_price_eur` into `property_context`
  - Agents use this value as a “fair value anchor” for setting max_willing_eur

Components
1) Property Documents

Location: data/raw_properties/
Format: markdown
Two layers of information:
    Structured header: key:value pairs used for metadata extraction
    Unstructured body: narrative description used for RAG retrieval and rationale grounding

2) Metadata Extraction

Implemented in m2_index.py:
- Scans the first N lines (header area)
  - Maps common keys into a canonical schema:\
  `price_eur, size_sqm, rooms, neighborhood, property_type`
Converts numeric fields to integers safely (handles commas in € values, “sqm”, etc.)
This metadata becomes the shared “truth” used by:
- RAG retrieval display
- ML dataset creation
- Auction decision logic

3) Vector Database (Chroma)

Stores embedded chunks + metadata
Used for:
- semantic search in chat
- retrieving full context for an auction property by property_id
Persistence:
- CHROMA_DIR directory on disk

4) RAG Chat

Implemented in m3_chat_cli.py:
- Maintains small conversational state:
  - recent message history
  - optional preferences summary (budget, bedrooms, locations)

On each user message:
- update preference summary (small LLM call)
- combine summary + user message for retrieval query
- retrieve top chunks
- answer using retrieved context only (grounded)

This supports follow-up questions like:
- “Which is best by price/space?”
- “Does it have parking?”
- “What about public transport nearby?”

5) Multi-Agent Auction System

**Agents**
BuyerAgent represents one bidder with:
- budget_eur
- strategy (aggressive / conservative / analytical)
- preferences (neighborhoods, rooms, legal constraints, etc.)

Each agent uses the LLM to produce a structured BidResponse:
- decision: bid or pass
- bid amount
- rationale grounded in property context
- optional max_willing_eur cap

**Orchestrator**
AuctionOrchestrator runs a discrete-round auction:
Parameters:
- min_increment_eur
- max_rounds
- optional shuffle of bidding order per round

Rules:
- Bid must exceed current highest by at least min_increment
- Agents cannot exceed their budget
- Agents do not raise their own bid when currently leading
- Auction ends when a full round produces no higher bid

Outputs:
- winner, final price
- full history of bids/passes and rationales

******ML Option A: Price Prediction Integration******

**Motivation**
Auction agents need an estimate of “fair value” to avoid irrational overbidding and to reason about discount vs. asking price.

**Model**
Type: Regression model (Ridge or similar)

Features:
- rooms, size_sqm, neighborhood, property_type

Training:
- uses cross-validation (small dataset)
- saves pipeline as models/price_model.joblib

Inference in Auction

Before running each auction:
- predict predicted_price_eur from the property’s metadata
- inject into property_context["ml"]["predicted_price_eur"]

Agent usage:
Agents set max_willing_eur as a function of:
- predicted fair value
- risk discounts (Act 15 / inheritance)
- strategy discount (conservative bids below fair value)
Hard validation prevents bids that exceed max willingness

This creates more realistic collisions and early termination when the auction price exceeds predicted value.

**Key Design Decisions & Trade-offs**

1. Chroma metadata as single source of truth
    Avoids parsing markdown multiple times for ML and auction logic
    Ensures retrieval, ML features, and agent reasoning stay consistent

2. CLI-first design 
   Quick to test and easy to run
   Focus on correctness and architecture rather than UI

3. Small dataset → Cross-validation
    With ~30 synthetic rows, single split metrics are unstable
    CV provides more reliable performance estimates

4. Deterministic auction rules + LLM decisions
   LLM produces rationale and suggested bid
   System enforces hard constraints (budget, min increment, max willingness)


