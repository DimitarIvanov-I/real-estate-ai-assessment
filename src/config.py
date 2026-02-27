import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
BASE_DIR = Path(__file__).resolve().parents[1]

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

RAW_PROPERTIES_DIR = BASE_DIR / "src" / "data" / "raw_properties"
CHROMA_DIR = BASE_DIR / "src" / "data" / "chroma_db"
COLLECTION_NAME = "properties"