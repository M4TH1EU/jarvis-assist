import json
import sqlite3
from typing import List, Dict

from homeassistant.helpers.llm import Tool

from .const import LOGGER
from .llamacpp_adapter import LlamaCppClient


class EmbeddingsDatabase:
    def __init__(self, llama_cpp: LlamaCppClient, db_path: str, overwrite: bool = False):
        self.conn = sqlite3.connect(db_path)
        self.llama_cpp = llama_cpp
        self._initialize_tables()

        self.overwrite = overwrite

        self.tools: List[Tool] = []
        self.entities: Dict[str, Dict] = {}

    def _initialize_tables(self):
        cur = self.conn.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS tool_embeddings (name TEXT PRIMARY KEY, description TEXT, vector TEXT)")
        cur.execute("CREATE TABLE IF NOT EXISTS entity_embeddings (name TEXT PRIMARY KEY, label TEXT, vector TEXT)")
        self.conn.commit()

    def close(self):
        self.conn.close()

    async def store_tool_embedding(self, tool: Tool):
        """Store or update tool embedding in the database."""
        if not tool in self.tools or self.overwrite:
            self.tools.append(tool)

            cur = self.conn.cursor()
            vec = (await self.llama_cpp.embeddings([tool.description]))[0]
            cur.execute("INSERT OR REPLACE INTO tool_embeddings VALUES (?, ?, ?)",
                        (tool.name, tool.description, json.dumps(vec)))
            self.conn.commit()

    async def store_tools_embeddings(self, tools: List[Tool]):
        for tool in tools:
            await self.store_tool_embedding(tool)

    async def store_entity_embedding(self, entity_id: str, entity: Dict):
        """Store or update entity embedding in the database."""
        if entity_id not in self.entities or self.overwrite:
            self.entities[entity_id] = entity

            cur = self.conn.cursor()
            label = entity.get("names", "")
            vec = (await self.llama_cpp.embeddings([label]))[0]
            cur.execute("INSERT OR REPLACE INTO entity_embeddings VALUES (?, ?, ?)",
                        (entity_id, label, json.dumps(vec)))
            self.conn.commit()

    async def store_entities_embeddings(self, entities: Dict):
        if not isinstance(entities, dict) or "entities" not in entities:
            LOGGER.warning("Invalid entities format, expected a dictionary with 'entities' key.")

        for entity_id, entity_data in entities["entities"].items():
            await self.store_entity_embedding(entity_id, entity_data)

    async def matching_entities(self, user_input: str | list, top_k=3) -> Dict[str, Dict]:
        """Find matching entities based on user input and return the top_k matches for the self.entities dictionary."""
        cur = self.conn.cursor()
        cur.execute("SELECT name, label, vector FROM entity_embeddings")
        rows = cur.fetchall()

        if isinstance(user_input, str):
            user_input = (await self.llama_cpp.embeddings([user_input]))[0]

        matches = []

        for name, label, vector in rows:
            vec = json.loads(vector)
            similarity = _cosine_similarity(user_input, vec)
            matches.append((name, label, similarity))

        # Sort by similarity and take the top_k
        matches.sort(key=lambda x: x[2], reverse=True)
        top_matches = matches[:top_k]

        return {
            name: self.entities.get(name, {"name": name, "label": label})
            for name, label, _ in top_matches
        }

    async def matching_tools(self, user_input: str | list, top_k=3) -> List[Tool]:
        """Find matching tools based on user input and return the top_k matches."""
        cur = self.conn.cursor()
        cur.execute("SELECT name, description, vector FROM tool_embeddings")
        rows = cur.fetchall()

        if isinstance(user_input, str):
            user_input = (await self.llama_cpp.embeddings([user_input]))[0]

        matches = []

        for name, description, vector in rows:
            vec = json.loads(vector)
            similarity = _cosine_similarity(user_input, vec)
            matches.append((name, similarity))

        # Sort by similarity and take the top_k
        matches.sort(key=lambda x: x[1], reverse=True)
        top_matches = matches[:top_k]

        tools_dict = {tool.name: tool for tool in self.tools}
        return [
            tools_dict.get(name)
            for name, _ in top_matches
            if tools_dict.get(name) is not None
        ]


def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm_a = sum(a ** 2 for a in vec1) ** 0.5
    norm_b = sum(b ** 2 for b in vec2) ** 0.5
    return dot_product / (norm_a * norm_b) if norm_a and norm_b else 0.0
