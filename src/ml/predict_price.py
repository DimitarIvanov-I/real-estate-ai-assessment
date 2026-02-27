from pathlib import Path
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = Path(__file__).resolve().parent / "models" / "price_model.joblib"

class PricePredictor:
    def __init__(self):
        self.model = joblib.load(MODEL_PATH)

    def predict(self, meta: dict) -> int | None:
        try:
            rooms = meta.get("rooms")
            size = meta.get("size_sqm")
            neighborhood = meta.get("neighborhood", "unknown")

            # model was trained on "property_type"
            property_type = meta.get("property_type") or meta.get("type") or "unknown"

            if rooms is None or size is None:
                return None

            X = pd.DataFrame([{
                "rooms": int(rooms),
                "size_sqm": int(size),
                "neighborhood": str(neighborhood),
                "property_type": str(property_type),
            }])

            pred = float(self.model.predict(X)[0])
            return int(max(pred, 0))

        except Exception as e:
            # TEMP debug (remove later if you want)
            print(f"[PricePredictor] predict failed: {type(e).__name__}: {e}")
            return None