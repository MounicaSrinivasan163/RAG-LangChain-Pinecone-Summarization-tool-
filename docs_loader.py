import csv
import os

DOC_REGISTRY_FILE = "indexed_documents.csv"

def save_documents(doc_name: str, doc_id: str):
    rows = []

    # Load existing
    if os.path.exists(DOC_REGISTRY_FILE):
        with open(DOC_REGISTRY_FILE, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if any(r["doc_id"] == doc_id for r in rows):
            return

    # Append new
    rows.append({"doc_name": doc_name, "doc_id": doc_id})

    # Write back
    with open(DOC_REGISTRY_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["doc_name", "doc_id"])
        writer.writeheader()
        writer.writerows(rows)

def load_documents() -> dict:
    if not os.path.exists(DOC_REGISTRY_FILE):
        return {}
    with open(DOC_REGISTRY_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {row["doc_name"]: row["doc_id"] for row in reader}

