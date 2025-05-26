"""The Llama Assist integration."""

from __future__ import annotations

import asyncio
import logging

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform, CONF_URL
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryNotReady
from homeassistant.helpers import llm

from llm import LlamaAssistAPI
from .const import DOMAIN, DEFAULT_TIMEOUT, LLAMA_LLM_API, CONF_BLACKLIST_TOOLS
from .llamacpp_adapter import LlamaCppClient

LOGGER = logging.getLogger(__name__)
PLATFORMS = (Platform.CONVERSATION,)


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up the Llama Assist integration."""
    settings = {**entry.data, **entry.options}

    if not any([x.id == LLAMA_LLM_API for x in llm.async_get_apis(hass)]):
        llm.async_register_api(hass, LlamaAssistAPI(hass))

    client = LlamaCppClient(base_url=settings.get(CONF_URL), hass=hass,
                            blacklist_tools=settings.get(CONF_BLACKLIST_TOOLS, []))
    try:
        async with asyncio.timeout(DEFAULT_TIMEOUT):
            await client.health()
    except TimeoutError as err:
        raise ConfigEntryNotReady(err) from err

    hass.data.setdefault(DOMAIN, {})[entry.entry_id] = {
        "client": client
    }

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload Ollama."""
    if not await hass.config_entries.async_unload_platforms(entry, PLATFORMS):
        return False

    hass.data[DOMAIN].pop(entry.entry_id)
    return True
