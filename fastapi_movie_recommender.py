import os
import time
import hashlib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler, normalize
from rapidfuzz import fuzz, process

# ✅ FastAPI App
app = FastAPI(title="Hybrid Movie Recommendation API", version="2.0", description="Semantic + Fuzzy Movie Recommender")

# ✅ Config
file_path = r"C:\Users\adith\Downloads\tmdb_movies_cleaned.csv"
model_name = "sentence-transformers/all-mpnet-base-v2"

# ✅ Load Model
model = SentenceTransformer(model_name)

# ✅ Load and Preprocess Dataset
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")

df = pd.read_csv(file_path, encoding="ISO-8859-1")
df["Text_Combined"] = (
    df["Movie Title"].fillna("").astype(str) + " " +
    df["Cleaned_Keywords"].fillna("").astype(str).apply(lambda x: (x + " ") * 3) +
    df["Description"].fillna("").astype(str).apply(lambda x: (x + " ") * 2)
)

# ✅ Generate a hash for text content to manage cache
def generate_text_hash(text_list):
    joined = "".join(text_list)
    return hashlib.md5(joined.encode("utf-8")).hexdigest()

text_data = df["Text_Combined"].fillna("").tolist()
text_hash = generate_text_hash(text_data)
embedding_file = f"tmdb_embeddings_{model_name.split('/')[-1]}_{text_hash}.npy"

# ✅ Load or Generate Embeddings
if os.path.exists(embedding_file) and os.path.getsize(embedding_file) > 0:
    embeddings = np.load(embedding_file)
    if embeddings.shape[0] != len(df):
        print("⚠️ Dataset changed! Regenerating embeddings...")
        os.remove(embedding_file)

if not os.path.exists(embedding_file):
    print("⏳ Generating SBERT embeddings (768D)...")
    embeddings = model.encode(text_data, batch_size=32, show_progress_bar=True)
    np.save(embedding_file, embeddings)
    print(f"✅ Embeddings saved: {embedding_file}")

# ✅ Normalize embeddings for faster cosine similarity
embeddings = normalize(embeddings)

# ✅ API Input Schema
class QueryRequest(BaseModel):
    query: str
    top_n: int = 10

# ✅ Hybrid Movie Finder Function
def find_similar_movies(query: str, top_n: int = 10):
    timings = {}
    t0 = time.time()

    # ✅ 1. Exact Match
    exact = df[df["Movie Title"].str.lower() == query.lower()]
    timings["Exact Match"] = time.time() - t0
    if not exact.empty:
        return [{
            "rank": 1,
            "movie_title": exact.iloc[0]["Movie Title"],
            "score": 1.0,
            "match_type": "Exact"
        }]

    # ✅ 2. Semantic Search
    t1 = time.time()
    query_embedding = normalize(model.encode([query])[0].reshape(1, -1))
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    timings["Semantic"] = time.time() - t1

    # ✅ Rescale similarities to (0.7 - 1.0)
    scaler = MinMaxScaler((0.7, 1.0))
    similarities_scaled = scaler.fit_transform(similarities.reshape(-1, 1)).flatten()

    # ✅ Top-N Semantic Results
    top_indices = np.argsort(similarities_scaled)[::-1][:top_n]
    results = [
        {
            "rank": i + 1,
            "movie_title": df.iloc[idx]["Movie Title"],
            "score": round(float(similarities_scaled[idx]), 6),
            "match_type": "Semantic"
        }
        for i, idx in enumerate(top_indices)
    ]
    timings["Top-N"] = time.time() - t1

    # ✅ 3. Fuzzy fallback
    if results and results[0]["score"] < 0.80:
        t3 = time.time()
        match = process.extractOne(query, df["Movie Title"].tolist(), scorer=fuzz.token_sort_ratio)
        timings["Fuzzy"] = time.time() - t3
        if match and match[1] > 85:
            return [{
                "rank": 1,
                "movie_title": match[0],
                "score": 0.99,
                "match_type": "Fuzzy"
            }]

    timings["Total"] = time.time() - t0
    print("⏱ Timing Breakdown:", {k: f"{v:.3f}s" for k, v in timings.items()})
    return results

# ✅ API Endpoints
@app.get("/")
def home():
    return {"message": "Welcome to the Hybrid Movie Recommendation API!"}

@app.post("/recommend")
def recommend_movies(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    recommendations = find_similar_movies(request.query, request.top_n)
    if not recommendations:
        raise HTTPException(status_code=404, detail="No recommendations found.")
    
    return {"query": request.query, "recommendations": recommendations}

# ✅ Debug Endpoint: Check if Description is used
@app.get("/check-description")
def check_description(index: int = 0):
    if index < 0 or index >= len(df):
        raise HTTPException(status_code=404, detail="Index out of range")
    
    row = df.iloc[index]
    return {
        "movie_title": row["Movie Title"],
        "description_used": row["Description"],
        "text_combined": row["Text_Combined"]
    }
