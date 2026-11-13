import hashlib
import os
import re

import httpx
import numpy as np

_STOPWORDS = set("""a an the and or of to in on for with is are was be by at as it this that from
after before since about our your their what which who how when do does did can i we you
""".split())
_TOKEN = re.compile(r"[a-z0-9]+")


class EmbeddingError(RuntimeError):
    pass


class HashingEmbedder:
    """Keyword baseline: bag-of-words feature hashing. Deterministic, used by the tests."""
    name = "hashing"
    min_score = 0.2

    def __init__(self, dim=1024):
        self.dim = dim

    def _tokens(self, text):
        toks = [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]
        return [t[:-1] if t.endswith("s") and len(t) > 3 else t for t in toks]

    def embed(self, text):
        vec = np.zeros(self.dim, dtype=np.float32)
        for tok in self._tokens(text):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec

    def embed_document(self, text):
        return self.embed(text)

    def embed_query(self, text):
        return self.embed(text)


class OllamaEmbedder:
    """Semantic embeddings from an open-weight model served by Ollama (default nomic-embed-text)."""

    def __init__(self, model=None, base_url=None, timeout=60.0):
        self.model = model or os.getenv("EMBED_MODEL", "nomic-embed-text")
        self.base_url = (base_url or os.getenv("OLLAMA_URL", "http://localhost:11434")).rstrip("/")
        self.timeout = timeout
        self.name = f"ollama:{self.model}"
        self.min_score = float(os.getenv("EMBED_MIN_SCORE", "0.63"))
        self.use_prefixes = "nomic" in self.model  # nomic models expect task prefixes

    def _embed(self, text):
        try:
            r = httpx.post(f"{self.base_url}/api/embed", json={"model": self.model, "input": [text]},
                           timeout=self.timeout)
        except httpx.HTTPError as e:
            raise EmbeddingError(f"cannot reach Ollama embeddings at {self.base_url}: {e}") from e
        if r.status_code != 200:
            raise EmbeddingError(f"Ollama embed returned HTTP {r.status_code}: {r.text[:200]}")
        try:
            vec = np.asarray(r.json()["embeddings"][0], dtype=np.float32)
        except (ValueError, KeyError, IndexError, TypeError) as e:
            raise EmbeddingError("malformed embedding response from Ollama") from e
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec  # unit length, so dot product == cosine similarity

    def embed_document(self, text):
        return self._embed(f"search_document: {text}" if self.use_prefixes else text)

    def embed_query(self, text):
        return self._embed(f"search_query: {text}" if self.use_prefixes else text)


def make_embedder(kind=None):
    kind = (kind or os.getenv("EMBEDDER", "ollama")).lower()
    if kind == "hashing":
        return HashingEmbedder()
    if kind == "ollama":
        return OllamaEmbedder()
    raise ValueError(f"unknown EMBEDDER {kind!r} (use 'ollama' or 'hashing')")