# 🏠 Real Estate Agents Simulation

## Overview

This project implements a simplified real estate marketplace simulation that includes:

- 📄 Property ingestion from unstructured Markdown documents  
- 🔎 RAG-based property search using vector embeddings (Chroma + OpenAI)  
- 🤖 Multi-agent auction simulation (LLM-driven buyer agents)  
- 📈 ML-based price prediction model (Bonus: Option A)

All components are implemented in **Python** and run via CLI.

---

# Requirements

- Python 3.10+
- OpenAI API key

---

#  Setup Instructions

## 1. Clone the repository

```
git clone https://github.com/DimitarIvanov-I/real-estate-ai-assessment.git
cd tech_assessment
```

## 2. Install dependencies

```pip install -r requirements.txt```

## 3. Configure environment variables

```export OPENAI_API_KEY="your_api_key_here"```

## HOW TO RUN

**Step 0 — Build Raw Demo Properties**
`python -m src.m1_generate_docs`

**Step 1 — Build Vector Index (RAG)**
Processes Markdown files, extracts metadata, chunks text, and builds the Chroma index.
`python -m src.m2_index`

**Step 2 — Run Property Search Chat**
`python -m src.m3_chat_cli`

**Step 3 — Build ML Dataset**
`python -m src.ml.build_dataset`

**Step 4 — Train Price Prediction Model**
`python -m src.ml.train_price_model`

**Step 5 — Run Auction Simulation**
`python -m src.agents.simulate_auction`# real-estate-ai-assessment
