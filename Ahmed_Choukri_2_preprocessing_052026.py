"""
01_preprocessing.py
Puls-Events - Step 2: Loading, filtering and structuring OpenAgenda data
Target regions: Ile-de-France, Hauts-de-France
Period: events from last 12 months + upcoming events
"""

import pandas as pd
import numpy as np
import re
import json
import os
from datetime import datetime, timezone, timedelta
from tqdm import tqdm

# Parameters
CSV_PATH        = "evenements-publics-openagenda.csv"
OUTPUT_PATH     = "data_processed.json"
TARGET_REGIONS  = ["Ile-de-France", "Hauts-de-France"]
DATE_LIMIT      = datetime.now(timezone.utc) - timedelta(days=365)

# Relevant columns
COLS = [
    "Identifiant",
    "Titre",
    "Description",
    "Description longue",
    "Mots cles",
    "Premiere date - Debut",
    "Premiere date - Fin",
    "Derniere date - Debut",
    "Derniere date - Fin",
    "Nom du lieu",
    "Adresse",
    "Code postal",
    "Ville",
    "Departement",
    "Region",
    "Pays",
    "Categorie",
    "URL canonique",
    "Evenement physique ou en ligne",
    "Etat de l'evenement",
    "Age minimum",
    "Age maximum",
]


def clean_html(text: str) -> str:
    """Removes HTML tags and cleans whitespace."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_date(value) -> datetime | None:
    """Parses ISO 8601 date to UTC datetime, returns None if invalid."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        dt = datetime.fromisoformat(value.strip())
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None


def build_document_text(row: pd.Series) -> str:
    """Builds raw text that will be vectorized."""
    parts = []
    if row.get("Titre"):
        parts.append(f"Event: {row['Titre']}")
    if row.get("Categorie"):
        parts.append(f"Category: {row['Categorie']}")
    if row.get("Description"):
        parts.append(f"Description: {row['Description']}")
    if row.get("Description longue"):
        parts.append(f"Details: {row['Description longue']}")
    if row.get("Ville"):
        location = row["Ville"]
        if row.get("Departement"):
            location += f", {row['Departement']}"
        if row.get("Region"):
            location += f", {row['Region']}"
        parts.append(f"Location: {location}")
    if row.get("Adresse"):
        parts.append(f"Address: {row['Adresse']}")
    if row.get("date_start_fmt"):
        parts.append(f"Start: {row['date_start_fmt']}")
    if row.get("date_end_fmt"):
        parts.append(f"End: {row['date_end_fmt']}")
    if row.get("Mots cles"):
        parts.append(f"Keywords: {row['Mots cles']}")
    return "\n".join(parts)


def main():
    print("=" * 60)
    print("  PULS-EVENTS — OpenAgenda Preprocessing")
    print("=" * 60)

    # 1. Load data
    print(f"\nLoading {CSV_PATH} ...")
    df = pd.read_csv(
        CSV_PATH,
        sep=";",
        usecols=lambda c: c in COLS,
        low_memory=False,
        dtype=str,
    )
    print(f"   OK {len(df):,} events loaded — {len(df.columns)} columns")

    # 2. Region filter
    print(f"\nRegion filtering: {TARGET_REGIONS}")
    df = df[df["Region"].isin(TARGET_REGIONS)].copy()
    print(f"   OK {len(df):,} events after geographic filter")

    # 3. Date filter
    print(f"\nPeriod filtering (>= {DATE_LIMIT.strftime('%Y-%m-%d')}) ...")
    df["_date_end"] = df["Derniere date - Fin"].apply(parse_date)
    before_filter = len(df)
    # Keep: end >= date_limit OR unknown date (for upcoming events without end)
    df = df[
        df["_date_end"].isna() | (df["_date_end"] >= DATE_LIMIT)
    ].copy()
    print(f"   OK {len(df):,} events kept (removed: {before_filter - len(df):,})")

    # 4. Text cleaning
    print("\nCleaning text fields ...")
    for col in ["Description", "Description longue"]:
        if col in df.columns:
            df[col] = df[col].apply(clean_html)

    for col in ["Titre", "Ville", "Adresse", "Categorie", "Mots cles", "Region", "Departement"]:
        if col in df.columns:
            df[col] = df[col].fillna("").str.strip()

    # 5. Formatted readable dates
    df["date_start_fmt"] = df["Premiere date - Debut"].apply(
        lambda x: parse_date(x).strftime("%d/%m/%Y %H:%M") if parse_date(x) else ""
    )
    df["date_end_fmt"] = df["Derniere date - Fin"].apply(
        lambda x: parse_date(x).strftime("%d/%m/%Y %H:%M") if parse_date(x) else ""
    )

    # 6. Remove duplicates
    before = len(df)
    df = df.drop_duplicates(subset=["Identifiant"]).copy()
    print(f"   OK Duplicates removed: {before - len(df):,}")

    # 7. Remove rows without title or description
    df = df[
        (df["Titre"].str.len() > 0) | (df["Description"].str.len() > 0)
    ].copy()
    print(f"   OK {len(df):,} valid events after cleaning")

    # 8. Build document text for vectorization
    print("\nBuilding document texts ...")
    tqdm.pandas(desc="   Texts")
    df["document_text"] = df.progress_apply(build_document_text, axis=1)

    # 9. Export structured JSON
    print(f"\nExporting to {OUTPUT_PATH} ...")
    records = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="   Export"):
        records.append({
            "id":               str(row.get("Identifiant", "")),
            "title":            row.get("Titre", ""),
            "description":      row.get("Description", ""),
            "long_description": row.get("Description longue", ""),
            "category":         row.get("Categorie", ""),
            "keywords":         row.get("Mots cles", ""),
            "city":             row.get("Ville", ""),
            "department":       row.get("Departement", ""),
            "region":           row.get("Region", ""),
            "address":          row.get("Adresse", ""),
            "postal_code":      row.get("Code postal", ""),
            "date_start":       row.get("Premiere date - Debut", ""),
            "date_end":         row.get("Derniere date - Fin", ""),
            "date_start_fmt":   row.get("date_start_fmt", ""),
            "date_end_fmt":     row.get("date_end_fmt", ""),
            "url":              row.get("URL canonique", ""),
            "event_type":       row.get("Evenement physique ou en ligne", ""),
            "status":           row.get("Etat de l'evenement", ""),
            "age_min":          row.get("Age minimum", ""),
            "age_max":          row.get("Age maximum", ""),
            "document_text":    row.get("document_text", ""),
        })

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print(f"  Preprocessing completed")
    print(f"  {len(records):,} events exported to {OUTPUT_PATH}")
    print(f"  Regions: {', '.join(TARGET_REGIONS)}")
    print(f"  Period: from {DATE_LIMIT.strftime('%d/%m/%Y')}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()