import random
import yaml
from pathlib import Path
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from src.config import OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL, CHROMA_DIR, COLLECTION_NAME, RAW_PROPERTIES_DIR
from src.agents.schemas import AgentProfile
from src.agents.buyer_agent import BuyerAgent
from src.agents.orchestrator import AuctionOrchestrator
import json
from pathlib import Path
from src.ml.predict_price import PricePredictor


def save_result(result):
    Path("logs").mkdir(exist_ok=True)
    with open(f"logs/auction_{result.property_id}.json", "w") as f:
        json.dump(result.model_dump(), f, indent=2)

def load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def get_vectordb():
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model=OPENAI_EMBEDDING_MODEL)
    return Chroma(collection_name=COLLECTION_NAME, persist_directory=str(CHROMA_DIR), embedding_function=embeddings)

def list_property_ids() -> list[str]:
    # read filenames: property_001.md etc; metadata property_id is P001 etc
    files = sorted(Path(RAW_PROPERTIES_DIR).glob("property_*.md"))
    # map to Pxxx based on filename index
    ids = []
    for fp in files:
        stem = fp.stem  # property_001
        num = stem.split("_")[1]
        ids.append(f"P{num}")
    return ids

def retrieve_property_context(vectordb: Chroma, property_id: str, k: int = 10) -> dict:
    docs = vectordb.similarity_search(query=property_id, k=k, filter={"property_id": property_id})
    chunks = []
    meta = {}
    for d in docs:
        md = d.metadata or {}
        meta = md or meta
        chunks.append(d.page_content)
    return {
        "property_id": property_id,
        "metadata": meta,
        "chunks": chunks,
    }

def main():
    cfg = load_config()
    client = OpenAI(api_key=OPENAI_API_KEY)
    model = cfg["llm"]["model"]

    vectordb = get_vectordb()
    predictor = PricePredictor()

    agent_profiles = [AgentProfile(**a) for a in cfg["agents"]]
    agents = [BuyerAgent(p, client, model) for p in agent_profiles]

    orch = AuctionOrchestrator(
        agents=agents,
        min_increment_eur=cfg["auction"]["min_increment_eur"],
        max_rounds=cfg["auction"]["max_rounds"],
    )

    # pick N properties to auction
    all_ids = list_property_ids()
    chosen = random.sample(all_ids, k=min(cfg["auction"]["num_properties"], len(all_ids)))

    for pid in chosen:
        ctx = retrieve_property_context(vectordb, pid, k=12)

        # ---- ML integration ----
        pred = predictor.predict(ctx["metadata"])
        ctx["ml"] = {"predicted_price_eur": pred}

        print("META used for ML:", {
            "rooms": ctx["metadata"].get("rooms"),
            "size_sqm": ctx["metadata"].get("size_sqm"),
            "neighborhood": ctx["metadata"].get("neighborhood"),
            "type": ctx["metadata"].get("type"),
        })
        # ------------------------

        asking = ctx["metadata"].get("price_eur")
        if isinstance(asking, int) and asking > 0:
            starting_bid = int(asking * cfg["auction"]["starting_bid_pct"])
        else:
            starting_bid = 50000  # fallback

        result = orch.run_auction(ctx, starting_bid_eur=starting_bid)

        print("\n" + "=" * 80)
        print(f"PROPERTY: {pid} | Asking: €{asking} | Start: €{starting_bid}")
        pred = ctx.get("ml", {}).get("predicted_price_eur")
        print(f"ML predicted: €{pred}" if pred else "ML predicted: n/a")
        print(f"WINNER: {result.winner} | FINAL: €{result.final_price_eur}")
        print("-" * 80)

        # show last ~10 events
        for e in result.history[-10:]:
            print(f"R{e['round']:02d} {e['agent']}: {e['decision']} {e['bid_eur']} | {e['rationale']}")
        print("=" * 80)

        save_result(result)

if __name__ == "__main__":
    main()