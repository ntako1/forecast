from fastapi import FastAPI, HTTPException, Query
from typing import Optional

from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import subprocess
import os
import math

app = FastAPI(title="AI Model Runner & Data Viewer")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ή βάλε ["http://localhost:3000"] αν έχεις React app
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AI_SCRIPT = os.path.join(BASE_DIR, "ai.py")
DATA_FILE = os.path.join(BASE_DIR, "Greece_forecast_ai.xlsx")


# ==========================================================
# 🚀 Endpoint 1: Run AI model (trains / forecasts)
# ==========================================================
@app.get("/run_model")
def run_model():
    """Τρέχει το ai.py και επιστρέφει μήνυμα κατάστασης."""
    try:
        if not os.path.exists(AI_SCRIPT):
            raise HTTPException(status_code=404, detail="ai.py not found")

        print("🚀 Starting AI model retraining...")

        result = subprocess.run(
            ["python", AI_SCRIPT],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore"
        )

        if result.returncode != 0:
            print("❌ AI model error:", result.stderr)
            return JSONResponse(
                content={"status": "error", "message": result.stderr},
                status_code=500
            )

        print("✅ AI model finished successfully.")
        return {"status": "done", "message": "Model retrained successfully."}

    except Exception as e:
        print("💥 Exception while running AI:", e)
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================================
# 📊 Endpoint 2: Display data (read Excel as JSON)
# ==========================================================
@app.get("/display_data")
def display_data(
    scenario: Optional[str] = Query("all", description="Scenario filter: all | historic | mild | extreme"),
    year: Optional[int] = Query(None, description="Year filter (optional)")
):
    """Διαβάζει το Excel και επιστρέφει JSON με φίλτρα και μοναδικά historic δεδομένα."""
    if not os.path.exists(DATA_FILE):
        raise HTTPException(status_code=404, detail="Excel file not found. Run /run_model first!")

    try:
        print("📂 Reading Excel:", DATA_FILE)
        xls = pd.ExcelFile(DATA_FILE)
        print("📊 Sheets found:", xls.sheet_names)

        all_data = []
        historic_added = set()  # για να αποφύγουμε διπλά historic

        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name)
            df.columns = [str(c).strip() for c in df.columns]
            df = df.fillna(0)

            scen_col = next((c for c in df.columns if "scen" in c.lower()), None)
            if scen_col:
                df["Scenario"] = df[scen_col].apply(
                    lambda x: "historic" if x in [0, "0", "", None] else str(x).lower()
                )
            else:
                df["Scenario"] = "historic"

            for _, row in df.iterrows():
                try:
                    year_val = int(row["Year"]) if pd.notna(row["Year"]) else None
                    mag = float(row.get("Predicted_Magnitude") or row.get("Magnitude") or 0)
                    inj = float(row.get("AI_Predicted_Injuries") or row.get("No. Injured") or 0)
                    scen = str(row.get("Scenario", "historic")).lower()

                    if not year_val:
                        continue

                    # ✅ Αν είναι historic, κράτα το μόνο μία φορά ανά έτος
                    if scen == "historic":
                        if year_val in historic_added:
                            continue
                        historic_added.add(year_val)

                    all_data.append({
                        "year": year_val,
                        "magnitude": mag,
                        "injuries": inj,
                        "scenario": scen
                    })

                except Exception as e:
                    print("⚠️ Error parsing row:", e)
                    continue

        # Καθαρισμός NaN
        for d in all_data:
            if isinstance(d.get("magnitude"), float) and math.isnan(d["magnitude"]):
                d["magnitude"] = 0
            if isinstance(d.get("injuries"), float) and math.isnan(d["injuries"]):
                d["injuries"] = 0

        # 🎯 Εφαρμογή φίλτρων
        filtered_data = all_data
        if scenario and scenario.lower() != "all":
            filtered_data = [d for d in filtered_data if d["scenario"] == scenario.lower()]
        if year:
            filtered_data = [d for d in filtered_data if d["year"] == year]

        if not filtered_data:
            return {"records": 0, "data": [], "message": "No data found for given filters."}

        print(f"✅ Returning {len(filtered_data)} records (scenario={scenario}, year={year}).")
        return {"records": len(filtered_data), "data": filtered_data}

    except Exception as e:
        print("💥 Exception while reading Excel:", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/refresh_data")
def refresh_data():
    global df
    df = pd.read_excel("Greece_forecast_ai.xlsx")
    return {"status": "✅ Data reloaded"}
