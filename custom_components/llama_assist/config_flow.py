"""Config flow for Llama Assist integration."""
from __future__ import annotations

import asyncio
import logging
import sys
from typing import Any

import voluptuous as vol
from homeassistant.config_entries import ConfigFlow, ConfigEntry, OptionsFlow
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import llm
from homeassistant.helpers.selector import TextSelector, TextSelectorConfig, TextSelectorType, SelectOptionDict, \
    TemplateSelector, SelectSelector, SelectSelectorConfig, NumberSelector, NumberSelectorConfig, NumberSelectorMode

from . import LlamaCppClient, LlamaAssistAPI
from .const import DOMAIN, DEFAULT_TIMEOUT, CONF_PROMPT, CONF_MAX_HISTORY, DEFAULT_MAX_HISTORY, LLAMA_LLM_API, \
    DISABLE_REASONING, CONF_DISABLE_REASONING, EXISTING_TOOLS, CONF_BLACKLIST_TOOLS, CONF_USE_EMBEDDINGS_TOOLS, \
    USE_EMBEDDINGS_TOOLS, USE_EMBEDDINGS_ENTITIES, \
    CONF_USE_EMBEDDINGS_ENTITIES, CONF_SERVER_EMBEDDINGS_URL, CONF_COMPLETION_SERVER_URL
from .llamacpp_adapter import RequestError

_LOGGER = logging.getLogger(__name__)

STEP_USER_DATA_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_COMPLETION_SERVER_URL): TextSelector(
            TextSelectorConfig(type=TextSelectorType.URL)
        ),
        vol.Optional(CONF_SERVER_EMBEDDINGS_URL, default=""): TextSelector(
            TextSelectorConfig(type=TextSelectorType.URL)
        ),
    }
)


class LlamaAssistConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Llama Assist."""

    VERSION = 1

    def __init__(self) -> None:
        """Initialize config flow."""
        self.url: str | None = None
        self.url_embeddings: str | None = None
        self.client: LlamaCppClient | None = None

    async def async_step_user(self, user_input: dict[str, Any] | None = None):
        """Handle the initial step."""
        user_input = user_input or {}
        self.url = user_input.get(CONF_COMPLETION_SERVER_URL, self.url)
        self.url_embeddings = user_input.get(CONF_SERVER_EMBEDDINGS_URL, self.url_embeddings)

        if not any([x.id == LLAMA_LLM_API for x in llm.async_get_apis(self.hass)]):
            llm.async_register_api(self.hass, LlamaAssistAPI(self.hass))

        if self.url is None:
            return self.async_show_form(step_id="user", data_schema=STEP_USER_DATA_SCHEMA, last_step=False)

        errors = {}

        try:
            self.client = LlamaCppClient(base_url=self.url, embeddings_base_url=self.url_embeddings, hass=self.hass)

            async with asyncio.timeout(DEFAULT_TIMEOUT):
                await self.client.health()
        except TimeoutError | RequestError:
            errors["base"] = "cannot_connect"
        except Exception:
            _LOGGER.exception("Unexpected exception")
            errors["base"] = "unknown"

        if errors:
            return self.async_show_form(step_id="user", data_schema=STEP_USER_DATA_SCHEMA, errors=errors)

        return self.async_create_entry(title=f"Llama Assist ({self.url})", data={CONF_COMPLETION_SERVER_URL: self.url,
                                                                                 CONF_SERVER_EMBEDDINGS_URL: self.url_embeddings})

    @staticmethod
    @callback
    def async_get_options_flow(config_entry: ConfigEntry) -> OptionsFlow:
        """Create the options flow."""
        return LlamaAssistOptionsFlow(config_entry)


class LlamaAssistOptionsFlow(OptionsFlow):
    """Ollama options flow."""

    def __init__(self, config_entry: ConfigEntry) -> None:
        """Initialize options flow."""
        self.url: str = config_entry.data.get(CONF_COMPLETION_SERVER_URL, "")
        self.embeddings_url: str = config_entry.data.get(CONF_SERVER_EMBEDDINGS_URL, "")

    async def async_step_init(self, _user_input=None):
        """Manage the options."""
        return await self.async_step_user()

    async def async_step_user(self, user_input: dict[str, Any] | None = None):
        """Manage the options."""
        if user_input is not None:
            return self.async_create_entry(title=f"Llama Assist ({self.url})", data=user_input)

        settings = self.config_entry.data | self.config_entry.options

        schema = llama_assist_config_option_schema(self.hass, options=settings)
        return self.async_show_form(step_id="user", data_schema=vol.Schema(schema))


def llama_assist_config_option_schema(
        hass: HomeAssistant, options: dict[str, Any]
) -> dict:
    """Ollama options schema."""
    # hass_apis: list[SelectOptionDict] = []
    tools: list[SelectOptionDict] = []

    # api: AssistAPI
    # for api in llm.async_get_apis(hass):
    #     hass_apis.append(SelectOptionDict(
    #         label=api.name,
    #         value=api.id,
    #     ))

    for tool in EXISTING_TOOLS:
        tools.append(SelectOptionDict(
            label=tool,
            value=tool,
        ))

    return {
        vol.Required(
            CONF_PROMPT,
            description={
                "suggested_value": options.get(
                    CONF_PROMPT, llm.DEFAULT_INSTRUCTIONS_PROMPT
                )
            },
        ): TemplateSelector(),
        # vol.Optional(
        #     CONF_LLM_HASS_API,
        #     description={"suggested_value": options.get(CONF_LLM_HASS_API)},
        # ): SelectSelector(SelectSelectorConfig(options=hass_apis, multiple=True)),
        vol.Required(
            CONF_MAX_HISTORY,
            description={
                "suggested_value": options.get(CONF_MAX_HISTORY, DEFAULT_MAX_HISTORY)
            },
        ): NumberSelector(
            NumberSelectorConfig(
                min=0, max=sys.maxsize, step=1, mode=NumberSelectorMode.BOX
            )
        ),
        vol.Required(
            CONF_DISABLE_REASONING,
            default=options.get(CONF_DISABLE_REASONING, DISABLE_REASONING)
        ): bool,
        vol.Optional(
            CONF_SERVER_EMBEDDINGS_URL,
            description={
                "suggested_value": options.get(CONF_SERVER_EMBEDDINGS_URL, "")
            },
        ): TextSelector(
            TextSelectorConfig(type=TextSelectorType.URL)
        ),
        vol.Required(
            CONF_USE_EMBEDDINGS_TOOLS,
            default=options.get(CONF_USE_EMBEDDINGS_TOOLS, USE_EMBEDDINGS_TOOLS)
        ): bool,
        vol.Required(
            CONF_USE_EMBEDDINGS_ENTITIES,
            default=options.get(CONF_USE_EMBEDDINGS_ENTITIES, USE_EMBEDDINGS_ENTITIES)
        ): bool,
        vol.Required(
            CONF_BLACKLIST_TOOLS,
            description={
                "suggested_value": options.get(CONF_BLACKLIST_TOOLS, [])
            },
        ): SelectSelector(SelectSelectorConfig(options=tools, multiple=True)),
    }
