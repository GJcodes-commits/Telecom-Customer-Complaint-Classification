"""Download a manageable sample of public FCC telecom complaints."""
from pathlib import Path
import requests
import pandas as pd

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
OUTPUT = DATA_DIR / "fcc_complaints.csv"
URL = "https://opendata.fcc.gov/resource/3xyp-aqkj.csv"

def main():
    params = {
        "$select": "ticket_created,issue_type,method,issue,state",
        "$where": "issue_type in ('Phone','Internet','TV') AND issue IS NOT NULL",
        "$order": "ticket_created DESC",
        "$limit": 50000,
    }
    print("Connecting to the official FCC open-data API...")
    response = requests.get(URL, params=params, timeout=120)
    response.raise_for_status()
    OUTPUT.write_bytes(response.content)
    df = pd.read_csv(OUTPUT)
    if df.empty:
        raise RuntimeError("The FCC API returned no rows. Try again later or check the dataset endpoint.")
    print(f"Downloaded {len(df):,} records.")
    print(f"Saved to: {OUTPUT}")
    print("\nPreview:")
    print(df.head())
    print("\nService forms:")
    print(df["issue_type"].value_counts(dropna=False).head(10))

if __name__ == "__main__":
    main()
