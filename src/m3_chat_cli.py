import json
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma  # pip install -U langchain-chroma

from src.config import (
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OPENAI_EMBEDDING_MODEL,
    CHROMA_DIR,
    COLLECTION_NAME,
)

client = OpenAI(api_key=OPENAI_API_KEY)


@dataclass
class ChatState:
    history: List[Dict[str, str]] = field(default_factory=list)
    preferences_summary: str = ""
    last_recommended_ids: List[str] = field(default_factory=list)

    def add_user(self, msg: str):
        self.history.append({"role": "user", "content": msg})
        self.history = self.history[-10:]

    def add_assistant(self, msg: str):
        self.history.append({"role": "assistant", "content": msg})
        self.history = self.history[-10:]


def extract_property_ids(text: str) -> List[str]:
    ids = re.findall(r"\bP\d{3}\b", text)
    seen = set()
    out = []
    for x in ids:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def wants_details(msg: str) -> bool:
    m = msg.lower().strip()
    triggers = ["more details", "tell me more", "details", "describe", "expand", "elaborate"]
    if any(t in m for t in triggers):
        return True
    # short confirmations like "yes", "ok", "sure" often mean "go on"
    return m in {"yes", "y", "ok", "okay", "sure", "please", "go on", "continue"}


def get_vectordb():
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model=OPENAI_EMBEDDING_MODEL)
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )


def _doc_to_item(d) -> Dict[str, Any]:
    md = d.metadata or {}
    return {
        "property_id": md.get("property_id", "UNKNOWN"),
        "neighborhood": md.get("neighborhood", ""),
        "price_eur": md.get("price_eur", ""),
        "rooms": md.get("rooms", ""),
        "size_sqm": md.get("size_sqm", ""),
        "chunk": d.page_content,
    }


def retrieve_context(vectordb: Chroma, query: str, k: int = 10) -> List[Dict[str, Any]]:
    docs = vectordb.similarity_search(query, k=k)
    return [_doc_to_item(d) for d in docs]


def retrieve_for_property(vectordb: Chroma, property_id: str, k: int = 8) -> List[Dict[str, Any]]:
    docs = vectordb.similarity_search(
        query=property_id,
        k=k,
        filter={"property_id": property_id},
    )
    return [_doc_to_item(d) for d in docs]


def update_preferences_summary(state: ChatState, user_msg: str) -> str:
    prompt = f"""
You maintain a short user preference summary for a property search chatbot.

Current summary (may be empty):
{state.preferences_summary}

New user message:
{user_msg}

Update the summary in 1-3 lines. Include only stable constraints:
- city/neighborhood preferences
- budget / price ceiling
- rooms/bedrooms (note: dataset uses "rooms" field)
- key must-haves (metro, terrace, parking, furnished, etc.)
If the user changes something, overwrite the old constraint.
Return ONLY the updated summary text.
""".strip()

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "Output ONLY the updated summary text."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


def answer_with_rag(state: ChatState, user_msg: str, retrieved: List[Dict[str, Any]]) -> str:
    context_json = json.dumps(retrieved, ensure_ascii=False)

    prompt = f"""
You are a conversational property search assistant.

Use ONLY the provided retrieved context.
If something is not in the context, say you don't have enough information.

User preference summary:
{state.preferences_summary}

Recent conversation:
{json.dumps(state.history[-6:], ensure_ascii=False)}

Current user message:
{user_msg}

Retrieved context (JSON list). Each item contains:
property_id, neighborhood, price_eur, rooms, size_sqm, chunk.
{context_json}

Instructions:
- Understand the user's intent in context (comparisons, follow-ups, refinements).
- If the user refers to "this", "that", "the first one", "these", interpret using conversation context.
- Recommend relevant properties from the retrieved context only.
- Do NOT introduce properties not present in the context.
- If comparing, you may compute €/sqm if price_eur and size_sqm exist.
- Be concise but specific.
- Ask a short clarifying question only if truly needed.

Format:
1) Pxxx — Neighborhood — €price — X rooms
   - reason grounded in chunk
   - reason grounded in chunk

(Repeat if needed)
""".strip()

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "Be concise and grounded in the provided context."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
    )
    return resp.choices[0].message.content.strip()


def answer_details(state: ChatState, user_msg: str, retrieved: List[Dict[str, Any]], target_id: str) -> str:
    context_json = json.dumps(retrieved, ensure_ascii=False)

    prompt = f"""
You are a property assistant. Use ONLY the provided context.

The user is asking for MORE DETAILS about property {target_id}.
If info is missing, say what is missing.

Recent conversation:
{json.dumps(state.history[-8:], ensure_ascii=False)}

Property context (JSON chunks for {target_id}):
{context_json}

Write a compact "property card" with:
- Property ID, neighborhood, price, size, rooms
- Key features (bullets)
- Neighborhood & transport (metro/bus/tram if mentioned)
- Condition / amenities
- Inspection/legal notes
- Ask ONE natural follow-up question at the end (optional)

Do NOT list other properties.
""".strip()

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "Be specific and grounded. Do not introduce other properties."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()


def main():
    vectordb = get_vectordb()
    state = ChatState()

    print("🏠 Property Search Chat (type 'exit' to quit)")

    while True:
        user_msg = input("\nYou: ").strip()
        if not user_msg:
            continue
        if user_msg.lower() in ("exit", "quit"):
            break

        state.add_user(user_msg)

        # Update preference summary for better follow-ups
        state.preferences_summary = update_preferences_summary(state, user_msg)

        # If user says "yes / more details", drill into the most recently recommended property (if any)
        target_id: Optional[str] = state.last_recommended_ids[0] if state.last_recommended_ids else None

        if target_id and wants_details(user_msg):
            retrieved = retrieve_for_property(vectordb, target_id, k=10)
            if not retrieved:
                # fallback to general retrieval if filter returns nothing
                recent_history = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in state.history[-4:]])
                retrieval_query = f"""
User preference summary:
{state.preferences_summary}

Recent conversation:
{recent_history}

Current user message:
{user_msg}
""".strip()
                retrieved = retrieve_context(vectordb, retrieval_query, k=10)
            answer = answer_details(state, user_msg, retrieved, target_id)
        else:
            recent_history = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in state.history[-4:]])
            retrieval_query = f"""
User preference summary:
{state.preferences_summary}

Recent conversation:
{recent_history}

Current user message:
{user_msg}
""".strip()
            retrieved = retrieve_context(vectordb, retrieval_query, k=10)

            if not retrieved:
                print("\nBot: I couldn't find any relevant properties in the index yet. Try re-indexing.")
                continue

            answer = answer_with_rag(state, user_msg, retrieved)

        state.add_assistant(answer)
        print(f"\nBot:\n{answer}")

        # update shortlist from whatever IDs appear in the bot response
        ids = extract_property_ids(answer)
        if ids:
            state.last_recommended_ids = ids


if __name__ == "__main__":
    main()