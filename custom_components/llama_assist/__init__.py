"""The Llama Assist integration."""

from __future__ import annotations

from pathlib import Path

import openai
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryNotReady
from homeassistant.helpers import llm as ha_llm
from homeassistant.helpers.httpx_client import get_async_client
from posthog.ai.openai import AsyncOpenAI

from .const import DOMAIN, HEALTHCHECK_TIMEOUT, LLAMA_LLM_API, CONF_BLACKLIST_TOOLS, CONF_SERVER_EMBEDDINGS_URL, \
    CONF_COMPLETION_SERVER_URL, CONF_USE_EMBEDDINGS_TOOLS, USE_EMBEDDINGS_TOOLS, PLATFORMS, EMBEDDINGS_SQLITE, \
    OVERWRITE_EMBEDDINGS, CONF_OVERWRITE_EMBEDDINGS
from .embeddings import EmbeddingsDatabase
from .llm import LlamaAssistAPI

type LlamaAPIClientsConfigEntry = ConfigEntry[LlamaAPIClients]


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up the Llama Assist integration."""
    settings = {**entry.data, **entry.options}

    if not any([x.id == LLAMA_LLM_API for x in ha_llm.async_get_apis(hass)]):
        ha_llm.async_register_api(hass, LlamaAssistAPI(hass))

    completion_client = openai.AsyncOpenAI(
        api_key="none",
        base_url=settings.get(CONF_COMPLETION_SERVER_URL, ""),
        http_client=get_async_client(hass)
    )
    embeddings_client = openai.AsyncOpenAI(
        api_key="none",
        base_url=settings.get(CONF_SERVER_EMBEDDINGS_URL, ""),
        http_client=get_async_client(hass)
    ) if settings.get(CONF_SERVER_EMBEDDINGS_URL) else None

    try:
        await hass.async_add_executor_job(completion_client.with_options(timeout=10.0).models.list)
    except openai.OpenAIError as err:
        raise ConfigEntryNotReady(err) from err

    try:
        await hass.async_add_executor_job(
            embeddings_client.with_options(timeout=10.0).models.list) if embeddings_client else None
    except openai.OpenAIError as err:
        raise ConfigEntryNotReady(err) from err

    db_path = Path(hass.config.path(DOMAIN + "/" + EMBEDDINGS_SQLITE))
    db_path.parent.mkdir(parents=True, exist_ok=True)
    embeddings_db = EmbeddingsDatabase(
        embeddings_client,
        db_path=str(db_path),
        overwrite=settings.get(CONF_OVERWRITE_EMBEDDINGS, OVERWRITE_EMBEDDINGS),
        blacklist_tools=settings.get(CONF_BLACKLIST_TOOLS, []),
    )

    apis = LlamaAPIClients(
        completion_client=completion_client,
        embeddings_client=embeddings_client,
        embeddings_db=embeddings_db
    )

    entry.runtime_data = apis

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    return True


async def async_unload_entry(hass: HomeAssistant, entry: LlamaAPIClientsConfigEntry) -> bool:
    """Unload Ollama."""
    if not await hass.config_entries.async_unload_platforms(entry, PLATFORMS):
        return False

    hass.data[DOMAIN].pop(entry.entry_id)  # TODO: fix unload
    return True


class LlamaAPIClients:
    """A class to hold the API clients for Llama Assist."""

    def __init__(self, completion_client: AsyncOpenAI, embeddings_client: AsyncOpenAI | None = None,
                 embeddings_db: EmbeddingsDatabase | None = None):
        """Initialize the API clients."""
        self.completion_client = completion_client
        self.embeddings_client = embeddings_client
        self.embeddings_db = embeddings_db
