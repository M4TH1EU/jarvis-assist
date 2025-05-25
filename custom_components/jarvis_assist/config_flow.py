"""Config flow for Jarvis Assist integration."""
from __future__ import annotations

import asyncio
import logging
import sys
from typing import Any, Mapping

import voluptuous as vol
from homeassistant.config_entries import ConfigFlow, ConfigEntry, OptionsFlow
from homeassistant.const import CONF_URL, CONF_LLM_HASS_API
from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm
from homeassistant.helpers.selector import TextSelector, TextSelectorConfig, TextSelectorType, SelectOptionDict, \
    TemplateSelector, SelectSelector, SelectSelectorConfig, NumberSelector, NumberSelectorConfig, NumberSelectorMode

from . import LlamaCppClient, JarvisAssistAPI
from .const import DOMAIN, DEFAULT_TIMEOUT, CONF_PROMPT, CONF_MAX_HISTORY, DEFAULT_MAX_HISTORY, JARVIS_LLM_API

_LOGGER = logging.getLogger(__name__)

STEP_USER_DATA_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_URL): TextSelector(
            TextSelectorConfig(type=TextSelectorType.URL)
        ),
    }
)


class JarvisAssistConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Jarvis Assist."""

    VERSION = 1

    def __init__(self) -> None:
        """Initialize config flow."""
        self.url: str | None = None
        self.client: LlamaCppClient | None = None

    #
    # async def async_step_user(self, user_input: dict[str, Any] | None = None):
    #     """Handle the initial step."""
    #     if user_input is not None:
    #         server_url = user_input["server-url"]
    #         if not server_url.startswith("http://") and not server_url.startswith("https://"):
    #             return self.async_show_form(
    #                 step_id="user",
    #                 data_schema=DATA_SCHEMA,
    #                 errors={"base": "invalid_url"},
    #             )
    #
    #         await self.async_set_unique_id(server_url)
    #         self._abort_if_unique_id_configured()
    #
    #         return self.async_create_entry(
    #             title="Jarvis Assist",
    #             data={},
    #         )
    #
    #     return self.async_show_form(
    #         step_id="user",
    #         data_schema=DATA_SCHEMA,
    #     )

    async def async_step_user(self, user_input: dict[str, Any] | None = None):
        """Handle the initial step."""
        user_input = user_input or {}
        self.url = user_input.get(CONF_URL, self.url)

        if not any([x.id == JARVIS_LLM_API for x in llm.async_get_apis(self.hass)]):
            llm.async_register_api(self.hass, JarvisAssistAPI(self.hass))

        if self.url is None:
            return self.async_show_form(step_id="user", data_schema=STEP_USER_DATA_SCHEMA, last_step=False)

        errors = {}

        try:
            self.client = LlamaCppClient(base_url=self.url, hass=self.hass)

            async with asyncio.timeout(DEFAULT_TIMEOUT):
                await self.client.health()
        except TimeoutError:
            errors["base"] = "cannot_connect"
        except Exception:
            _LOGGER.exception("Unexpected exception")
            errors["base"] = "unknown"

        if errors:
            return self.async_show_form(step_id="user", data_schema=STEP_USER_DATA_SCHEMA, errors=errors)

        return self.async_create_entry(title=f"Jarvis Assist ({self.url})", data={CONF_URL: self.url})

    @staticmethod
    def async_get_options_flow(config_entry: ConfigEntry) -> OptionsFlow:
        """Create the options flow."""
        return JarvisAssistOptionsFlow(config_entry)


class JarvisAssistOptionsFlow(OptionsFlow):
    """Ollama options flow."""

    def __init__(self, config_entry: ConfigEntry) -> None:
        """Initialize options flow."""
        self.url: str = config_entry.data[CONF_URL]

    async def async_step_init(self, user_input: dict[str, Any] | None = None):
        """Manage the options."""
        if user_input is not None:
            return self.async_create_entry(title=f"Jarvis Assist ({self.url})", data=user_input)

        options: Mapping[str, Any] = self.config_entry.options or {}
        schema = jarvis_assist_config_option_schema(self.hass, options)
        return self.async_show_form(step_id="init", data_schema=vol.Schema(schema))


def jarvis_assist_config_option_schema(
        hass: HomeAssistant, options: Mapping[str, Any]
) -> dict:
    """Ollama options schema."""
    hass_apis: list[SelectOptionDict] = [
        SelectOptionDict(
            label=api.name,
            value=api.id,
        )
        for api in llm.async_get_apis(hass)
    ]

    return {
        vol.Optional(
            CONF_PROMPT,
            description={
                "suggested_value": options.get(
                    CONF_PROMPT, llm.DEFAULT_INSTRUCTIONS_PROMPT
                )
            },
        ): TemplateSelector(),
        vol.Optional(
            CONF_LLM_HASS_API,
            description={"suggested_value": options.get(CONF_LLM_HASS_API)},
        ): SelectSelector(SelectSelectorConfig(options=hass_apis, multiple=True)),
        vol.Optional(
            CONF_MAX_HISTORY,
            description={
                "suggested_value": options.get(CONF_MAX_HISTORY, DEFAULT_MAX_HISTORY)
            },
        ): NumberSelector(
            NumberSelectorConfig(
                min=0, max=sys.maxsize, step=1, mode=NumberSelectorMode.BOX
            )
        )
    }
