"""
02_vectorisation.py
Puls-Events - Step 3: Vectorization with Mistral Embeddings + FAISS Index
"""

import json
import os
import pickle
import faiss
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
from mistralai import Mistral
import time

# Configuration
load_dotenv()
MISTRAL_API_KEY  = os.getenv("MISTRAL_API_KEY")
JSON_PATH        = "data_processed.json"
FAISS_INDEX_PATH = "faiss_index.bin"
METADATA_PATH    = "faiss_metadata.pkl"
EMBED_MODEL      = "mistral-embed"
BATCH_SIZE       = 50
CHUNK_MAX_CHARS  = 2000


# 1. Chunking
def chunker(text: str, max_chars: int = CHUNK_MAX_CHARS) -> list[str]:
    """
    Splits text into chunks of max_chars characters,
    respecting sentence boundaries.
    """
    if len(text) <= max_chars:
        return [text]
    chunks = []
    while len(text) > max_chars:
        cut = text.rfind(". ", 0, max_chars)
        if cut == -1:
            cut = max_chars
        else:
            cut += 1
        chunks.append(text[:cut].strip())
        text = text[cut:].strip()
    if text:
        chunks.append(text)
    return chunks


# 2. Mistral Embeddings
def get_embeddings(client: Mistral, texts: list[str], retries: int = 5) -> list[list[float]]:
    """Mistral API call with automatic retry on rate limit."""
    for attempt in range(retries):
        try:
            response = client.embeddings.create(
                model=EMBED_MODEL,
                inputs=texts,
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            if "429" in str(e) or "capacity" in str(e).lower():
                wait = 10 * (attempt + 1)
                print(f"\n   Rate limit — waiting {wait}s (attempt {attempt+1}/{retries})")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Too many failed attempts on Mistral API.")


# Main
def main():
    print("=" * 60)
    print("  PULS-EVENTS — Vectorization & FAISS Index")
    print("=" * 60)

    # Load data
    print(f"\nLoading {JSON_PATH} ...")
    with open(JSON_PATH, encoding="utf-8") as f:
        events = json.load(f)
    print(f"   OK {len(events):,} events loaded")

    # Mistral client
    client = Mistral(api_key=MISTRAL_API_KEY)

    # Chunking all events
    print("\nSplitting into chunks ...")
    chunk_texts   = []
    chunk_metadata = []

    for evt in tqdm(events, desc="   Chunking"):
        text = evt.get("document_text", "").strip()
        if not text:
            continue
        pieces = chunker(text)
        for i, piece in enumerate(pieces):
            chunk_texts.append(piece)
            chunk_metadata.append({
                "chunk_id":    f"{evt['id']}_chunk{i}",
                "event_id":    evt["id"],
                "title":       evt.get("title", ""),
                "city":        evt.get("city", ""),
                "region":      evt.get("region", ""),
                "department":  evt.get("department", ""),
                "address":     evt.get("address", ""),
                "category":    evt.get("category", ""),
                "date_start":  evt.get("date_start_fmt", ""),
                "date_end":    evt.get("date_end_fmt", ""),
                "url":         evt.get("url", ""),
                "text":        piece,
            })

    print(f"   OK {len(chunk_texts):,} chunks generated from {len(events):,} events")

    # Batch vectorization
    print(f"\nVectorization with Mistral ({EMBED_MODEL}) ...")
    print(f"   Batch size: {BATCH_SIZE} — Number of batches: {len(chunk_texts) // BATCH_SIZE + 1}")

    all_embeddings = []
    for i in tqdm(range(0, len(chunk_texts), BATCH_SIZE), desc="   Embeddings"):
        batch = chunk_texts[i : i + BATCH_SIZE]
        embeddings = get_embeddings(client, batch)
        all_embeddings.extend(embeddings)
        time.sleep(1.5)

    print(f"   OK {len(all_embeddings):,} vectors generated")

    # Build FAISS index
    print("\nBuilding FAISS index ...")
    vectors = np.array(all_embeddings, dtype="float32")
    dimension = vectors.shape[1]
    print(f"   Vector dimension: {dimension}")

    # FlatL2 index (exact, optimal for < 100k vectors)
    index = faiss.IndexFlatL2(dimension)
    # Add ID mapping to retrieve chunks
    index_with_ids = faiss.IndexIDMap(index)
    ids = np.arange(len(vectors), dtype="int64")
    index_with_ids.add_with_ids(vectors, ids)

    print(f"   OK {index_with_ids.ntotal:,} vectors indexed in FAISS")

    # Save FAISS index
    faiss.write_index(index_with_ids, FAISS_INDEX_PATH)
    print(f"   OK Index saved: {FAISS_INDEX_PATH}")

    # Save metadata
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(chunk_metadata, f)
    print(f"   OK Metadata saved: {METADATA_PATH}")

    # Quick search test
    print("\nSemantic search test ...")
    query_test = "classical music concert in Paris"
    query_vector = get_embeddings(client, [query_test])[0]
    query_vector_np = np.array([query_vector], dtype="float32")

    distances, indices = index_with_ids.search(query_vector_np, k=3)

    print(f"\n   Query: \"{query_test}\"")
    print(f"   Top 3 results:")
    for rank, (idx, dist) in enumerate(zip(indices[0], distances[0]), 1):
        meta = chunk_metadata[idx]
        print(f"\n   [{rank}] L2 score: {dist:.4f}")
        print(f"       Title:    {meta['title']}")
        print(f"       City:     {meta['city']} ({meta['region']})")
        print(f"       Dates:    {meta['date_start']} -> {meta['date_end']}")
        print(f"       Category: {meta['category']}")

    print(f"\n{'=' * 60}")
    print(f"  Vectorization completed")
    print(f"  {len(all_embeddings):,} chunks indexed")
    print(f"  Files: {FAISS_INDEX_PATH} + {METADATA_PATH}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()