import os
import random
from openai import OpenAI
from src.config import OPENAI_API_KEY, OPENAI_MODEL, RAW_PROPERTIES_DIR

client = OpenAI(api_key=OPENAI_API_KEY)

NEIGHBORHOODS = [
    "Lozenets", "Ivan Vazov", "Mladost 1", "Mladost 4", "Studentski Grad",
    "Center", "Oborishte", "Krasno Selo", "Manastirski Livadi", "Druzhba 2",
    "Nadezhda", "Lyulin 5", "Geo Milev", "Hladilnika", "Beli Brezi",
]
PROPERTY_TYPES = ["apartment", "studio", "house", "penthouse"]
AMENITIES = [
    "metro access", "parking", "elevator", "storage room", "south exposure",
    "renovated bathroom", "new kitchen", "quiet street", "park nearby",
    "security entrance", "high ceilings", "terrace", "furnished",
]
LEGAL_NOTES = [
    "clear title", "pending inheritance documents", "mortgage-free",
    "Act 16 available", "Act 15 pending", "reconstructed building file ready",
]

def make_prompt(i: int) -> str:
    neighborhood = random.choice(NEIGHBORHOODS)
    prop_type = random.choice(PROPERTY_TYPES)
    rooms = random.choice([1, 2, 2, 3, 3, 4])
    size = random.randint(35, 140)
    amen = random.sample(AMENITIES, k=4)
    legal = random.choice(LEGAL_NOTES)
    base_price_per_sqm = {
        "center": 3200,
        "oborishte": 3000,
        "ivan vazov": 2800,
        "mladost 1": 1900,
        "mladost 4": 2100,
        "geo milev": 2200,
        "druzhba 2": 1700,
        "nadezhda": 1500,
        "studentski grad": 2000,
        "beli brezi": 2400,
        "manastirski livadi": 2300,
        "lozenets": 3100,
        'hladilnika': 2100,
        'lyulin 5': 1200,
        "krasno selo": 2600,
    }

    price = size * base_price_per_sqm[neighborhood.lower()]

    # small random noise
    price *= random.uniform(0.9, 1.1)

    # legal risk discount
    if "Act 15" in legal:
        price *= 0.92

    if "pending inheritance" in legal:
        price *= 0.85

    price = int(price)

    return f"""
Generate ONE realistic Sofia real-estate listing as Markdown.
It must be unstructured and rich in text (not just a table). Keep it plausible.

Include:
- A short structured header block with: Property ID, City, Neighborhood, Type, Price EUR, Size sqm, Rooms
- Then these sections with headings:
  ## Description (2-4 paragraphs)
  ## Neighborhood & Transport (1-2 paragraphs; mention metro/bus/tram realistically)
  ## Amenities & Condition (bullets allowed but include sentences too)
  ## Inspection Notes or Legal Notes (include {legal} and 1-2 more realistic notes)

Constraints:
- Property ID: P{i:03d}
- City must be Sofia
- Neighborhood must be {neighborhood}
- Type must be {prop_type}
- Price must be {price} EUR
- Size must be {size} sqm
- Rooms must be {rooms}
- Must naturally mention these amenities somewhere: {", ".join(amen)}
- No hallucinated real addresses; use general descriptions.

Return only the Markdown document.
""".strip()

def main(n: int = 30):
    os.makedirs(RAW_PROPERTIES_DIR, exist_ok=True)

    for i in range(1, n + 1):
        prompt = make_prompt(i)
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You generate realistic Bulgarian real estate listings in English."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.8,
        )
        text = resp.choices[0].message.content.strip()

        path = os.path.join(RAW_PROPERTIES_DIR, f"property_{i:03d}.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text + "\n")

        print(f"Wrote {path}")

if __name__ == "__main__":
    main(30)