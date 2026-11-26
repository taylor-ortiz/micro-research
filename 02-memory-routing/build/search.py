"""The shared retrieval method: hybrid BM25 + embedding search, top 10.

Every setup that searches uses `hybrid_search` with the user's message as the
query. Rankings from BM25 and from cosine similarity of OpenAI
text-embedding-3-small vectors are merged with reciprocal rank fusion (k = 60).
No LLM takes part.

Embeddings are cached in data/embeddings/ keyed by a hash of the text, so each
text is embedded once. The OpenAI key is added to requests by the session's
network proxy; nothing here reads or stores a key.
"""

import hashlib
import json
import math
import re
from collections import Counter

import httpx
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from . import config

EMBED_MODEL = "text-embedding-3-small"
EMBED_URL = "https://api.openai.com/v1/embeddings"
CACHE = config.DATA / "embeddings"
TOP_K = 10
RRF_K = 60
BM25_K1, BM25_B = 1.5, 0.75

_vectors: dict[str, np.ndarray] = {}


def _key(text):
    return hashlib.sha1(text.encode()).hexdigest()


def _load_cache():
    if _vectors or not (CACHE / "index.json").exists():
        return
    keys = json.loads((CACHE / "index.json").read_text())
    mat = np.load(CACHE / "vectors.npy")
    _vectors.update({k: mat[i] for i, k in enumerate(keys)})


def _save_cache():
    CACHE.mkdir(parents=True, exist_ok=True)
    keys = list(_vectors)
    (CACHE / "index.json").write_text(json.dumps(keys))
    np.save(CACHE / "vectors.npy", np.stack([_vectors[k] for k in keys]).astype(np.float32))


def embed(texts):
    """Vectors for texts (unit length), fetching only those not cached."""
    _load_cache()
    missing = sorted({t for t in texts if _key(t) not in _vectors})
    for i in range(0, len(missing), 256):
        chunk = missing[i:i + 256]
        r = httpx.post(EMBED_URL, timeout=120,
                       headers={"Authorization": "Bearer proxy-injected"},
                       json={"model": EMBED_MODEL, "input": chunk})
        r.raise_for_status()
        for t, d in zip(chunk, r.json()["data"]):
            v = np.array(d["embedding"], dtype=np.float32)
            _vectors[_key(t)] = v / np.linalg.norm(v)
    if missing:
        _save_cache()
    return np.stack([_vectors[_key(t)] for t in texts])


def _tokens(text):
    return [w for w in re.findall(r"[a-z0-9_]+", text.lower()) if w not in ENGLISH_STOP_WORDS]


def bm25_scores(query, docs):
    toks = [_tokens(d) for d in docs]
    n = len(docs)
    avg = sum(map(len, toks)) / max(n, 1)
    df = Counter(w for t in toks for w in set(t))
    q = _tokens(query)
    scores = []
    for t in toks:
        tf = Counter(t)
        s = 0.0
        for w in q:
            if w not in tf:
                continue
            idf = math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
            s += idf * tf[w] * (BM25_K1 + 1) / (tf[w] + BM25_K1 * (1 - BM25_B + BM25_B * len(t) / avg))
        scores.append(s)
    return np.array(scores)


def _ranks(scores):
    """1 = best. Tied scores share the best rank of their group, so list order
    (the store's shuffle) never breaks ties."""
    s = np.round(scores, 9)
    return np.array([int((s > x).sum()) + 1 for x in s])


def hybrid_search(query, memories, k=TOP_K):
    """memories: list of {"id", "memory"}. Returns (top_k_memories, full_ranking).

    full_ranking is every memory's id with its fused score and both component
    ranks, best first, for logging and for recall at other cutoffs later.
    """
    if not memories:
        return [], []
    docs = [m["memory"] for m in memories]
    bm = bm25_scores(query, docs)
    qv = embed([query])[0]
    cos = embed(docs) @ qv
    rb, re_ = _ranks(bm), _ranks(cos)
    fused = 1 / (RRF_K + rb) + 1 / (RRF_K + re_)
    # Ties in the fused score break by a hash of the id, never by list position.
    order = sorted(range(len(memories)), key=lambda i: (-fused[i], _key(memories[i]["id"])))
    ranking = [{"id": memories[i]["id"], "score": round(float(fused[i]), 6),
                "bm25_rank": int(rb[i]), "embed_rank": int(re_[i])} for i in order]
    return [memories[i] for i in order[:k]], ranking
