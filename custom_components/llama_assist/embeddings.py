import asyncio
from dataclasses import dataclass
from typing import List, Dict, Any

import chromadb
from chromadb import Settings
from homeassistant.core import HomeAssistant
from homeassistant.helpers.llm import Tool

from const import CHROMADB_PATH


@dataclass
class LlamaChroma:
    client: chromadb.Client
    tools_coll: chromadb.Collection
    entities_coll: chromadb.Collection


async def create_chroma(hass: HomeAssistant, entry_id: str) -> LlamaChroma:
    """Create a ChromaDB client with a unique path for each entry."""

    def _create_chroma_client():
        """Create a ChromaDB client."""
        return chromadb.Client(Settings(
            persist_directory=CHROMADB_PATH + "_" + entry_id,
            anonymized_telemetry=False
        ))

    client = await hass.async_add_executor_job(_create_chroma_client)

    return LlamaChroma(
        client=client,
        tools_coll=client.get_or_create_collection("tools"),
        entities_coll=client.get_or_create_collection("entities"),
    )


async def _ingest_tools(llama_chroma: LlamaChroma, tools: List[Tool]):
    ids = []
    texts = []
    metadatas = []
    for t in tools:
        ids.append(t.name)
        texts.append(t.description)
        metadatas.append({"tool_name": t.name})

    await asyncio.to_thread(
        llama_chroma.tools_coll.upsert,
        ids=ids,
        embeddings=None,
        metadatas=metadatas,
        documents=texts,
        images=None,
        uris=None
    )


async def _ingest_entities(llama_chroma: LlamaChroma, entities: Dict[str, Dict[str, Any]]):
    ids = []
    texts = []
    metadatas = []

    for ent_id, props in entities["entities"].items():
        ids.append(ent_id)
        texts.append(props["names"])
        metadatas.append(props)

    await asyncio.to_thread(
        llama_chroma.entities_coll.upsert,
        ids=ids,
        embeddings=None,
        metadatas=metadatas,
        documents=texts,
        images=None,
        uris=None
    )


async def get_matching_tools(
        llama_chroma: LlamaChroma,
        user_input: str,
        tools: List[Tool],
        count: int = 3,
) -> List[Tool]:
    await _ingest_tools(llama_chroma, tools)

    results = await asyncio.to_thread(
        llama_chroma.tools_coll.query,
        query_texts=[user_input],
        n_results=count,
        query_embeddings=None,
        query_images=None,
        query_uris=None,
        ids=None,
        where_document=None,
        where=None,
        include=["metadatas", "documents", "distances"],
    )

    matching_tool_names = results["ids"][0] if results["ids"] else []

    return [tool for tool in tools if tool.name in matching_tool_names]


async def get_matching_entities(
        llama_chroma: LlamaChroma,
        user_input: str,
        entities: dict,
        count: int = 3,
) -> dict:
    await _ingest_entities(llama_chroma, entities)

    results = await asyncio.to_thread(
        llama_chroma.entities_coll.query,
        query_texts=[user_input],
        n_results=count,
        query_embeddings=None,
        query_images=None,
        query_uris=None,
        ids=None,
        where_document=None,
        where=None,
        include=["metadatas", "documents", "distances"],
    )

    matching_entity_ids = results["ids"][0] if results["ids"] else []

    return {
        "entities": {
            ent_id: props
            for ent_id, props in entities["entities"].items()
            if ent_id in matching_entity_ids
        }
    }
