import re
from pathlib import Path
from pathlib import Path
import pandas as pd

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from src.config import (
    OPENAI_API_KEY,
    OPENAI_EMBEDDING_MODEL,
    CHROMA_DIR,
    COLLECTION_NAME,
)

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = Path("../data/raw_properties")
OUT_CSV = Path("data/ml/property_dataset.csv")


def get_vectordb() -> Chroma:
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model=OPENAI_EMBEDDING_MODEL)
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )


def build():
    vectordb = get_vectordb()

    # Pull raw records from the underlying Chroma collection
    col = vectordb._collection
    data = col.get(include=["metadatas"])

    metadatas = data.get("metadatas", []) or []
    if not metadatas:
        raise RuntimeError("No metadatas found in Chroma. Did you run m2_index.py first?")

    # many chunks per property -> dedupe by property_id
    by_pid = {}
    for md in metadatas:
        if not md:
            continue
        pid = md.get("property_id")
        if not pid:
            continue

        # Keep the first full record; later ones usually repeat
        if pid not in by_pid:
            by_pid[pid] = md

    rows = []
    for pid, md in sorted(by_pid.items()):
        row = {
            "property_id": pid,
            "city": md.get("city", "Sofia"),
            "neighborhood": md.get("neighborhood", "unknown"),
            "property_type": md.get("type", "unknown"),  # key is "type"
            "rooms": md.get("rooms"),
            "size_sqm": md.get("size_sqm"),
            "price_eur": md.get("price_eur"),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Keep only rows with target + required numeric features
    df = df.dropna(subset=["price_eur", "rooms", "size_sqm"]).copy()
    df["rooms"] = df["rooms"].astype(int)
    df["size_sqm"] = df["size_sqm"].astype(int)
    df["price_eur"] = df["price_eur"].astype(int)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    print(f"Saved dataset: {OUT_CSV} | rows={len(df)}")
    print(df.head(5).to_string(index=False))


if __name__ == "__main__":
    build()