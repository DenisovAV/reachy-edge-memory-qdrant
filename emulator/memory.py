"""What was said: the conversation, after it leaves the model's context.

The language model holds only the last few exchanges (demo/conversation.py's
ConversationWindow). Older ones move here in batches, embedded with bge-small
and stored in the robot's `memory` shard — the same Qdrant Edge shard as the
frames (emulator/frame_memory.py), told apart by `kind`. The model reaches
them only by asking (the `remember` tool), and the search is by meaning: "what
did I tell you about my dog?" finds "My dog is called Rex" with no word in
common.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Iterable

    import numpy as np
    from fastembed import TextEmbedding

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

# A conversation EXCHANGE, stored as "Person: ... — Reachy: ..." when it falls
# out of the language model's context window. The whole exchange rather than
# the person's line alone: measured on bge-small (8 exchanges, 11 questions),
# the right exchange ranked first 11/11 stored this way against 7/11 for the
# person's line — "what's my name?" is much closer to "Nice to meet you,
# Sasha!" than to "Hi Reachy, I'm Sasha."
EXCHANGE_KIND = "exchange"

# The gate for an exchange to count as recalled. Measured on the same set:
# right answers scored 0.636-0.781, the best wrong exchange for a specific
# question 0.665, and a question whose answer was never stored at most 0.620.
EXCHANGE_RECALL_MIN_SCORE = 0.62

# Loading the embedding model costs real wall time. Every TextMemory (and the
# knowledge base) in one process shares the one loaded model.
_embedder_cache: dict[str, "TextEmbedding"] = {}


def _embedder(model_name: str) -> "TextEmbedding":
    from fastembed import TextEmbedding  # lazy: only paths that need memory pay for it
    cached = _embedder_cache.get(model_name)
    if cached is None:
        cached = TextEmbedding(model_name)
        _embedder_cache[model_name] = cached
    return cached


class TextEmbedder(Protocol):
    """What TextMemory needs from an embedder: fastembed's embed/query_embed
    split — bge trains its query and document sides differently.
    demo/embed_client.py's RemoteBgeEmbedder has the same shape over
    HTTP, for a robot that does not load the model itself."""

    def embed(self, texts: list[str]) -> "Iterable[np.ndarray]":
        """Document-side vectors — what gets stored."""

    def query_embed(self, texts: list[str]) -> "Iterable[np.ndarray]":
        """Query-side vectors — what a search is done with."""


class TextMemory:
    """The stored conversation, searched by meaning.

    `store` is the shared EdgeStore with a `text` vector — the robot's
    `memory` shard; without one this memory makes a shard of its own at
    `path` (None: an ephemeral one, for tests)."""

    def __init__(self, path: str | None = None, *,
                 embedder: "TextEmbedder | None" = None,
                 model_name: str = DEFAULT_MODEL, store=None) -> None:
        from emulator.edge_store import KIND, TEXT, EdgeStore, match_value

        # An injected embedder is used as-is, and fastembed is never loaded:
        # the robot hands in a RemoteBgeEmbedder so this constructor never
        # imports it at all.
        self._embedder = embedder if embedder is not None else _embedder(model_name)
        # The size comes from a real embedding, so a different model can
        # never silently mismatch the shard it was created with.
        sample = next(iter(self._embedder.embed(["_"])))
        if store is None:
            store = EdgeStore(path, vectors={TEXT: len(sample)}, tenant_field=KIND)
        elif store.size(TEXT) not in (None, len(sample)):
            raise ValueError(f"the shard's text vector is {store.size(TEXT)}-d, "
                             f"the embedder gives {len(sample)}")
        self._store = store
        self._mine = match_value(KIND, EXCHANGE_KIND)
        # What is already stored, so the same exchange flushed twice (a
        # shutdown after an eviction) is not stored twice. Seeded from the
        # shard: it outlives the process.
        self._seen: set[str] = {
            payload.get("text") for _id, payload in
            self._store.iter_points(self._mine, fields=["text"])}

    def remember(self, text: str, kind: str = EXCHANGE_KIND,
                 meta: dict | None = None) -> None:
        if kind != EXCHANGE_KIND:
            raise ValueError(f"this memory stores exchanges, not {kind!r}")
        if text in self._seen:
            return
        from emulator.edge_store import TEXT

        vector = next(iter(self._embedder.embed([text])))
        self._store.add(self._store.new_id(), {TEXT: [float(x) for x in vector]},
                        {"text": text, "kind": kind, **(meta or {})})
        self._seen.add(text)

    def recall_exchanges(self, query: str, k: int = 3) -> list[dict]:
        """Stored exchanges nearest to `query`, best first, as
        `{text, kind, said_at, score}`; anything under
        EXCHANGE_RECALL_MIN_SCORE is dropped."""
        from emulator.edge_store import TEXT

        vector = next(iter(self._embedder.query_embed([query])))
        hits = self._store.search([float(x) for x in vector], k, self._mine,
                                  using=TEXT)
        return [hit for hit in hits if hit["score"] >= EXCHANGE_RECALL_MIN_SCORE]

    def latest_exchanges(self, n: int = 4) -> list[str]:
        """The last `n` stored exchanges, oldest first — what a question with
        nothing specific in it ("what did we talk about?") is answered from."""
        rows = [payload for _id, payload in self._store.iter_points(
            self._mine, fields=["text", "said_at"])]
        rows.sort(key=lambda row: row.get("said_at", 0.0))
        return [row["text"] for row in rows[-n:] if row.get("text")]

    def score_texts(self, query: str, texts: list[str]) -> list[float]:
        """Cosine similarity of `query` to each of `texts` in this memory's
        embedding space, storing nothing. The exchanges still inside the
        model's context are searched this way next to the stored ones
        (demo/conversation.py), so both share one scale and one gate — the
        same cosine the Edge shard computes for a stored point."""
        import numpy as np

        if not texts:
            return []
        query_vec = np.asarray(next(iter(self._embedder.query_embed([query]))),
                               dtype=np.float32)
        docs = np.asarray([np.asarray(v, dtype=np.float32)
                           for v in self._embedder.embed(list(texts))])
        query_vec /= max(float(np.linalg.norm(query_vec)), 1e-12)
        docs /= np.maximum(np.linalg.norm(docs, axis=1, keepdims=True), 1e-12)
        return [float(score) for score in docs @ query_vec]

    def count(self) -> int:
        """How many exchanges are stored — from the set kept current by
        remember(), so the dashboard's counter costs no shard query."""
        return len(self._seen)

    def close(self) -> None:
        """Put everything on disk and release the shard."""
        self._store.close()
