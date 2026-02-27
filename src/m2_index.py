import os
import re
from typing import Dict, List, Tuple

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from src.config import (
    OPENAI_API_KEY,
    OPENAI_EMBEDDING_MODEL,
    RAW_PROPERTIES_DIR,
    CHROMA_DIR,
    COLLECTION_NAME,
)

import re
from typing import Dict

KEY_MAP = {
    "property id": "property_id",
    "city": "city",
    "neighborhood": "neighborhood",
    "type": "property_type",
    "price eur": "price_eur",
    "price": "price_eur",
    "size sqm": "size_sqm",
    "size": "size_sqm",
    "rooms": "rooms",
    "bedrooms": "rooms",  # sometimes LLM uses bedrooms
}

LINE_RE = re.compile(r"^\s*(?:[-*]\s*)?(?P<k>[A-Za-z ]+?)\s*:\s*(?P<v>.+?)\s*$")

def _to_int(v: str):
    v = v.strip()
    v = v.replace("EUR", "").replace("sqm", "").replace(",", "").strip()
    # keep digits only
    digits = re.findall(r"\d+", v)
    return int(digits[0]) if digits else None

def extract_metadata(text: str, source_path: str) -> Dict:
    """
    Robust metadata extraction:
    - Scan first ~25 lines only (header area)
    - Accept bullet lines, bold markdown, etc.
    - Map common keys to canonical metadata fields
    """
    md = {"source": source_path}

    lines = text.splitlines()[:25]
    for line in lines:
        # remove markdown bold markers like "**City:** Sofia"
        line = line.replace("**", "").strip()
        m = LINE_RE.match(line)
        if not m:
            continue

        key = m.group("k").strip().lower()
        val = m.group("v").strip()

        if key in KEY_MAP:
            canon = KEY_MAP[key]
            if canon in ("price_eur", "size_sqm", "rooms"):
                num = _to_int(val)
                if num is not None:
                    md[canon] = num
            else:
                md[canon] = val

    return md

def load_documents() -> List:
    loader = DirectoryLoader(
        str(RAW_PROPERTIES_DIR),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    docs = loader.load()

    # attach metadata parsed from the document text
    for d in docs:
        d.metadata.update(extract_metadata(d.page_content, d.metadata.get("source", "")))
    return docs

def chunk_documents(docs: List) -> List:
    """
    Chunking strategy:
    - RecursiveCharacterTextSplitter keeps paragraphs/sections together better than naive splitting.
    - chunk_size ~ 900 chars with overlap ~ 120 to preserve context across boundaries.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=120,
        separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(docs)

def build_index(chunks: List):
    # reset old index (simple, deterministic rebuild)
    if os.path.exists(CHROMA_DIR):
        # Chroma is a directory; easiest is to delete and recreate
        import shutil
        shutil.rmtree(CHROMA_DIR)

    os.makedirs(CHROMA_DIR, exist_ok=True)

    embeddings = OpenAIEmbeddings(
        api_key=OPENAI_API_KEY,
        model=OPENAI_EMBEDDING_MODEL,
    )

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
    )
    return vectordb

def main():
    print("Loading documents...")
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")

    print("Chunking documents...")
    chunks = chunk_documents(docs)
    print(f"Created {len(chunks)} chunks")

    print("Building vector index (Chroma)...")
    build_index(chunks)
    print(f"Index built and persisted to: {CHROMA_DIR}")

if __name__ == "__main__":
    main()