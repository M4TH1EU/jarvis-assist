import json
import logging
import math
import sqlite3
from dataclasses import dataclass
from typing import List, Dict, Union

from homeassistant.helpers.llm import Tool
from openai import AsyncOpenAI
from openai.types import CreateEmbeddingResponse

from .const import EMBEDDINGS_MIN_SCORE

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class EntityRecord:
    """Simple container for an entity embedding record."""
    entity_id: str
    label: str
    vector: List[float]


@dataclass(frozen=True)
class ToolRecord:
    """Simple container for a tool embedding record."""
    name: str
    description: str
    vector: List[float]


class EmbeddingsDatabase:
    """
    Manage embeddings for tools and entities in a SQLite database.

    Usage:
        async with EmbeddingsDatabase(api, "/path/to/db.sqlite", overwrite=False) as db:
            await db.store_tools([tool1, tool2, ...])
            matches = await db.matching_tools("Turn on the TV")
    """

    def __init__(
            self,
            api: AsyncOpenAI,
            db_path: str,
            overwrite: bool = False,
            blacklist_tools: List[str] = None
    ) -> None:
        self.db_path = db_path
        self.api = api
        self.overwrite = overwrite
        self.blacklist_tools = blacklist_tools or []

        self._existing_tools: List[str] = []
        self._existing_entities: List[str] = []

        # In-memory caches
        # self._tools_cache: Dict[str, Tool] = {}
        # self._entities_cache: Dict[str, Dict] = {}
        self._all_tools: Dict[str, Tool] = {}
        self._all_entities: Dict[str, Dict] = {}

        self.conn = sqlite3.connect(self.db_path, timeout=10, isolation_level=None)
        self._configure_pragmas()
        self._create_tables()

        self._load_existing_tools()
        self._load_existing_entities()

    def _configure_pragmas(self) -> None:
        """Enable WAL and tune synchronous for a good speed/reliability trade-off."""
        cur = self.conn.cursor()
        cur.execute("PRAGMA journal_mode = WAL;")
        cur.execute("PRAGMA synchronous = NORMAL;")
        cur.close()

    def _create_tables(self) -> None:
        """Create embeddings tables if they do not exist."""
        with self.conn:
            self.conn.execute("""
                              CREATE TABLE IF NOT EXISTS tool_embeddings
                              (
                                  name        TEXT PRIMARY KEY,
                                  description TEXT NOT NULL,
                                  vector      TEXT NOT NULL
                              )
                              """)

            self.conn.execute("""
                              CREATE TABLE IF NOT EXISTS entity_embeddings
                              (
                                  name   TEXT PRIMARY KEY,
                                  label  TEXT NOT NULL,
                                  vector TEXT NOT NULL
                              )
                              """)

    def _load_existing_tools(self) -> None:
        """Load existing tools from the database into the cache."""
        cur = self.conn.cursor()
        cur.execute("SELECT name FROM tool_embeddings")
        for (name,) in cur.fetchall():
            if name not in self.blacklist_tools and name not in self._existing_tools:
                self._existing_tools.append(name)

    def _load_existing_entities(self) -> None:
        """Load existing entities from the database into the cache."""
        cur = self.conn.cursor()
        cur.execute("SELECT name FROM entity_embeddings")
        for (name,) in cur.fetchall():
            if name not in self._existing_entities:
                self._existing_entities.append(name)

    def close(self) -> None:
        """Close underlying SQLite connection."""
        self.conn.close()

    async def __aenter__(self) -> "EmbeddingsDatabase":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ──────────────────────────────────────────────────────
    #  Storage (batching for performance)
    # ──────────────────────────────────────────────────────

    async def store_tools(self, tools: List[Tool]) -> None:
        """
        Batch-store embeddings for a list of tools.
        Only new tools (or all if overwrite=True) will be embedded.
        """
        new_tools = [
            t for t in tools
            if self.overwrite or t.name not in self._existing_tools
        ]
        self._all_tools.update({t.name: t for t in tools})
        if not new_tools:
            return

        # Batch request descriptions → embeddings
        descriptions = [t.description for t in new_tools]
        try:
            vectors = await self._ensure_embedding(descriptions)
        except Exception as e:
            LOGGER.error("Failed to fetch embeddings for tools: %s", e)
            return

        records = [
            ToolRecord(name=tool.name, description=tool.description, vector=vec)
            for tool, vec in zip(new_tools, vectors)
        ]
        self._bulk_insert_tool_records(records)
        for t in new_tools:
            if t.name not in self._existing_tools:
                self._existing_tools.append(t.name)

    def _bulk_insert_tool_records(self, records: List[ToolRecord]) -> None:
        """Insert or replace multiple ToolRecords in one transaction."""
        sql = "INSERT OR REPLACE INTO tool_embeddings (name, description, vector) VALUES (?, ?, ?)"
        params = [(r.name, r.description, json.dumps(r.vector)) for r in records]
        try:
            with self.conn:
                self.conn.executemany(sql, params)
        except sqlite3.DatabaseError as db_err:
            LOGGER.exception("Error writing tool embeddings to DB: %s", db_err)

    async def store_entities(self, entities: Dict[str, Dict]) -> None:
        """
        Batch-store embeddings for an entities dict of the form:
            { "entities": { entity_id: { "names": label, ... }, ... } }
        """
        if not isinstance(entities, dict) or "entities" not in entities:
            LOGGER.warning("Invalid entities payload, expected a dict with 'entities' key.")
            return

        ent_map = entities["entities"]
        new_ids = [
            eid for eid in ent_map
            if self.overwrite or eid not in self._existing_entities
        ]
        self._all_entities.update(ent_map)
        if not new_ids:
            return

        labels = [ent_map[eid].get("names", "") for eid in new_ids]
        try:
            vectors = await self._ensure_embedding(labels)
        except Exception as e:
            LOGGER.error("Failed to fetch embeddings for entities: %s", e)
            return

        records = [
            EntityRecord(entity_id=eid, label=label, vector=vec)
            for eid, label, vec in zip(new_ids, labels, vectors)
        ]
        self._bulk_insert_entity_records(records)
        for eid in new_ids:
            if eid not in self._existing_entities:
                self._existing_entities.append(eid)

    def _bulk_insert_entity_records(self, records: List[EntityRecord]) -> None:
        """Insert or replace multiple EntityRecords in one transaction."""
        sql = "INSERT OR REPLACE INTO entity_embeddings (name, label, vector) VALUES (?, ?, ?)"
        params = [(r.entity_id, r.label, json.dumps(r.vector)) for r in records]
        try:
            with self.conn:
                self.conn.executemany(sql, params)
        except sqlite3.DatabaseError as db_err:
            LOGGER.exception("Error writing entity embeddings to DB: %s", db_err)

    # ──────────────────────────────────────────────────────
    #  Matching
    # ──────────────────────────────────────────────────────

    async def matching_tools(
            self,
            user_input: Union[str, List[float]],
            top_k: int = 3
    ) -> List[Tool]:
        """
        Return the top_k most similar Tool instances for the given user_input.
        If user_input is a str, fetch its embedding first.
        """
        query_vec = await self._ensure_embedding(user_input)
        rows = self.conn.execute(
            "SELECT name, vector FROM tool_embeddings"
        ).fetchall()

        scores: List[tuple[float, str]] = []
        for name, vec_json in rows:
            vec = json.loads(vec_json)
            scores.append((self._cosine_similarity(query_vec, vec), name))

        # Pick top_k
        top_names = [name for sim, name in sorted(scores, reverse=True)[:top_k] if sim >= EMBEDDINGS_MIN_SCORE]
        return [self._all_tools[name] for name in top_names if name in self._all_tools]

    async def matching_entities(
            self,
            user_input: Union[str, List[float]],
            top_k: int = 3
    ) -> Dict[str, Dict]:
        """
        Return a dict of the top_k most similar entities for the given user_input.
        Values come from the original entities payload cached in memory.
        """
        query_vec = await self._ensure_embedding(user_input)
        rows = self.conn.execute(
            "SELECT name, label, vector FROM entity_embeddings"
        ).fetchall()

        scores: List[tuple[float, str, str]] = []
        for name, label, vec_json in rows:
            vec = json.loads(vec_json)
            sim = self._cosine_similarity(query_vec, vec)
            scores.append((sim, name, label))

        top = sorted(scores, reverse=True)[:top_k]
        return {
            name: self._all_entities.get(name, {"name": name, "label": label})
            for sim, name, label in top if sim >= EMBEDDINGS_MIN_SCORE and name in self._all_entities
        }

    async def _ensure_embedding(
            self,
            user_input: Union[str, List[float], List[str]]
    ) -> List[float]:
        """If given a str, call llama_cpp.embeddings; otherwise pass through."""
        if isinstance(user_input, str):
            try:
                input_vector = await self.api.embeddings.create(input=user_input, model="none", encoding_format="float")
                input_vector = input_vector[0].model_extra.get('embedding', [])
                return input_vector[0] if input_vector else []
            except Exception as e:
                LOGGER.error("Embedding lookup failed: %s", e)
                return []
        elif isinstance(user_input, list) and all(isinstance(x, str) for x in user_input):
            # If a list of strings, assume it's a batch input
            try:
                input_vectors: List[List[float]] = []

                input_vectors_responses: List[CreateEmbeddingResponse] = await self.api.embeddings.create(
                    input=user_input, model="none", encoding_format="float")
                for response in input_vectors_responses:
                    if response.model_extra.get('embedding'):
                        vec = response.model_extra['embedding']
                        if isinstance(vec, list) and len(vec) > 0:
                            input_vectors.append(vec[0])

                return input_vectors or []
            except Exception as e:
                LOGGER.error("Batch embedding lookup failed: %s", e)
                return []

        return user_input

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        """Compute cosine similarity, guarding against zero‐length vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0
