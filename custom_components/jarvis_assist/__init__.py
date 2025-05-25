"""The Jarvis Assist integration."""

from __future__ import annotations

import asyncio
import logging

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform, CONF_URL
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryNotReady
from homeassistant.helpers import llm
from homeassistant.helpers.llm import API, LLMContext, APIInstance

from .const import DOMAIN, DEFAULT_TIMEOUT, JARVIS_LLM_API
from .llamacpp_adapter import LlamaCppClient

LOGGER = logging.getLogger(__name__)
PLATFORMS = (Platform.CONVERSATION,)


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up the Jarvis Assist integration."""
    settings = {**entry.data, **entry.options}

    if not any([x.id == JARVIS_LLM_API for x in llm.async_get_apis(hass)]):
        llm.async_register_api(hass, JarvisAssistAPI(hass))

    client = LlamaCppClient(base_url=settings[CONF_URL], hass=hass)
    try:
        async with asyncio.timeout(DEFAULT_TIMEOUT):
            await client.health()
    except TimeoutError as err:
        raise ConfigEntryNotReady(err) from err

    hass.data.setdefault(DOMAIN, {})[entry.entry_id] = client

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload Ollama."""
    if not await hass.config_entries.async_unload_platforms(entry, PLATFORMS):
        return False

    hass.data[DOMAIN].pop(entry.entry_id)
    return True


class JarvisAssistAPI(API):
    """My own API for LLMs."""

    def __init__(self, hass: HomeAssistant) -> None:
        """Initialize the API."""
        super().__init__(
            hass=hass,
            id=JARVIS_LLM_API,
            name="Jarvis Assist API",
        )

    async def async_get_api_instance(self, llm_context: LLMContext) -> APIInstance:
        """Return the instance of the API."""
        return APIInstance(
            api=self,
            api_prompt="Call the tools to fetch data from Home Assistant.",
            llm_context=llm_context,
            tools=[],
        )

#
# async def async_setup_api(hass: HomeAssistant, entry: ConfigEntry) -> None:
#     """Register the API with Home Assistant."""
#     # If the API is associated with a Config Entry, the LLM API must be
#     # unregistered when the config entry is unloaded.
#     unreg = llm.async_register_api(
#         hass,
#         JarvisAssistAPI(hass)
#     )
#     entry.async_on_unload(unreg)
