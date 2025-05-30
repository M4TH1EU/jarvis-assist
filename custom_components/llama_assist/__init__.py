"""The Llama Assist integration."""

from __future__ import annotations

import asyncio
from pathlib import Path

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryNotReady
from homeassistant.helpers import llm as ha_llm

from .const import DOMAIN, SERVER_API_TIMEOUT, LLAMA_LLM_API, CONF_BLACKLIST_TOOLS, CONF_SERVER_EMBEDDINGS_URL, \
    CONF_COMPLETION_SERVER_URL, CONF_USE_EMBEDDINGS_TOOLS, USE_EMBEDDINGS_TOOLS, PLATFORMS, EMBEDDINGS_SQLITE, \
    OVERWRITE_EMBEDDINGS, CONF_OVERWRITE_EMBEDDINGS
from .embeddings import EmbeddingsDatabase
from .llamacpp_adapter import LlamaCppClient
from .llm import LlamaAssistAPI


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up the Llama Assist integration."""
    settings = {**entry.data, **entry.options}

    if not any([x.id == LLAMA_LLM_API for x in ha_llm.async_get_apis(hass)]):
        ha_llm.async_register_api(hass, LlamaAssistAPI(hass))

    client = LlamaCppClient(base_url=settings.get(CONF_COMPLETION_SERVER_URL, ""),
                            embeddings_base_url=settings.get(CONF_SERVER_EMBEDDINGS_URL), hass=hass)
    try:
        async with asyncio.timeout(SERVER_API_TIMEOUT):
            await client.health()
    except TimeoutError as err:
        raise ConfigEntryNotReady(err) from err

    hass.data.setdefault(DOMAIN, {})[entry.entry_id] = {
        "client": client
    }

    db_path = Path(hass.config.path(DOMAIN + "/" + EMBEDDINGS_SQLITE))
    db_path.parent.mkdir(parents=True, exist_ok=True)
    hass.data[DOMAIN]["embeddings_db"] = EmbeddingsDatabase(
        client,
        db_path=str(db_path),
        overwrite=settings.get(CONF_OVERWRITE_EMBEDDINGS, OVERWRITE_EMBEDDINGS),
        blacklist_tools=settings.get(CONF_BLACKLIST_TOOLS, []),
    )

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload Ollama."""
    if not await hass.config_entries.async_unload_platforms(entry, PLATFORMS):
        return False

    hass.data[DOMAIN].pop(entry.entry_id)
    return True
