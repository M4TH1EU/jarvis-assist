import asyncio
from dataclasses import dataclass
from typing import List, Dict, Any

import chromadb
from chromadb import Settings
from homeassistant.core import HomeAssistant
from homeassistant.helpers.llm import Tool

from const import CHROMADB_PATH, DOMAIN


@dataclass
class LlamaColls:
    tools_coll: chromadb.Collection
    entities_coll: chromadb.Collection


async def get_or_create_chroma_client(hass: HomeAssistant) -> chromadb.Client:
    """Get the ChromaDB client from Home Assistant data."""
    if "chroma" not in hass.data.get(DOMAIN, {}):
        hass.data.setdefault(DOMAIN, {})["chroma"] = await _create_chroma(hass)

    return hass.data[DOMAIN]["chroma"]


async def _create_chroma(hass: HomeAssistant) -> chromadb.Client:
    """Create a ChromaDB client with a unique path for each entry."""

    def _create_chroma_client():
        """Create a ChromaDB client."""
        return chromadb.Client(Settings(
            persist_directory=CHROMADB_PATH,
            anonymized_telemetry=False
        ))

    return await hass.async_add_executor_job(_create_chroma_client)


async def get_or_create_collections(client: chromadb.Client, entry_id: str) -> LlamaColls:
    """Get or create collections for tools and entities."""
    return LlamaColls(
        tools_coll=client.get_or_create_collection("tools_" + entry_id),
        entities_coll=client.get_or_create_collection("entities_" + entry_id),
    )


async def _ingest_tools(llama_colls: LlamaColls, tools: List[Tool]):
    ids = []
    texts = []
    metadatas = []
    for t in tools:
        ids.append(t.name)
        texts.append(t.description)
        metadatas.append({"tool_name": t.name})

    await asyncio.to_thread(
        llama_colls.tools_coll.upsert,
        ids=ids,
        embeddings=None,
        metadatas=metadatas,
        documents=texts,
        images=None,
        uris=None
    )


async def _ingest_entities(llama_colls: LlamaColls, entities: Dict[str, Dict[str, Any]]):
    ids = []
    texts = []
    metadatas = []

    for ent_id, props in entities["entities"].items():
        ids.append(ent_id)
        texts.append(props["names"])
        metadatas.append(props)

    await asyncio.to_thread(
        llama_colls.entities_coll.upsert,
        ids=ids,
        embeddings=None,
        metadatas=metadatas,
        documents=texts,
        images=None,
        uris=None
    )


async def get_matching_tools(
        llama_colls: LlamaColls,
        user_input: str,
        tools: List[Tool],
        count: int = 3,
) -> List[Tool]:
    await _ingest_tools(llama_colls, tools)

    results = await asyncio.to_thread(
        llama_colls.tools_coll.query,
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
        llama_colls: LlamaColls,
        user_input: str,
        entities: dict,
        count: int = 3,
) -> dict:
    await _ingest_entities(llama_colls, entities)

    results = await asyncio.to_thread(
        llama_colls.entities_coll.query,
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
