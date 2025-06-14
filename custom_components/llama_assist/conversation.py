import json
from collections.abc import AsyncGenerator, Callable
from typing import Any, Literal, cast

import openai
from homeassistant.components import assist_pipeline, conversation
from homeassistant.components.conversation import AbstractConversationAgent, ConversationEntityFeature, SystemContent
from homeassistant.components.conversation import ConversationEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import device_registry as dr, intent, llm
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.util import yaml as yaml_util
from openai import AsyncOpenAI
from openai._streaming import AsyncStream
from openai.types.responses import (
    EasyInputMessageParam,
    FunctionToolParam,
    ResponseCompletedEvent,
    ResponseErrorEvent,
    ResponseFailedEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseFunctionToolCall,
    ResponseFunctionToolCallParam,
    ResponseIncompleteEvent,
    ResponseInputParam,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseOutputMessage,
    ResponseOutputMessageParam,
    ResponseReasoningItem,
    ResponseReasoningItemParam,
    ResponseStreamEvent,
    ResponseTextDeltaEvent,
)
from openai.types.responses.response_input_param import FunctionCallOutput
from voluptuous_openapi import convert

from . import DOMAIN, LlamaAssistAPI, LlamaAPIClientsConfigEntry
from .const import CONF_PROMPT, LLAMA_LLM_API, \
    CONF_USE_EMBEDDINGS_TOOLS, LOGGER, MAX_TOOL_ITERATIONS, CONF_USE_EMBEDDINGS_ENTITIES, ToolsEmbeddingsFeature, \
    EntitiesEmbeddingsFeature
from .embeddings import EmbeddingsDatabase


async def async_setup_entry(hass: HomeAssistant, config_entry: LlamaAPIClientsConfigEntry,
                            async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    """Set up conversation entities."""
    agent = LlamaConversationEntity(config_entry)
    async_add_entities([agent])


def _format_tool(
        tool: llm.Tool, custom_serializer: Callable[[Any], Any] | None
) -> FunctionToolParam:
    """Format tool specification."""
    return FunctionToolParam(
        type="function",
        name=tool.name,
        parameters=convert(tool.parameters, custom_serializer=custom_serializer),
        description=tool.description,
        strict=False,
    )


def _convert_content_to_param(
        content: conversation.Content,
) -> ResponseInputParam:
    """Convert any native chat message for this agent to the native format."""
    messages: ResponseInputParam = []
    if isinstance(content, conversation.ToolResultContent):
        return [
            FunctionCallOutput(
                type="function_call_output",
                call_id=content.tool_call_id,
                output=json.dumps(content.tool_result),
            )
        ]

    if content.content:
        role: Literal["user", "assistant", "system", "developer"] = content.role
        if role == "system":
            role = "developer"
        messages.append(
            EasyInputMessageParam(type="message", role=role, content=content.content)
        )

    if isinstance(content, conversation.AssistantContent) and content.tool_calls:
        messages.extend(
            ResponseFunctionToolCallParam(
                type="function_call",
                name=tool_call.tool_name,
                arguments=json.dumps(tool_call.tool_args),
                call_id=tool_call.id,
            )
            for tool_call in content.tool_calls
        )
    return messages


async def _transform_stream(
        chat_log: conversation.ChatLog,
        result: AsyncStream[ResponseStreamEvent],
        messages: ResponseInputParam,
) -> AsyncGenerator[conversation.AssistantContentDeltaDict]:
    """Transform an OpenAI delta stream into HA format."""
    async for event in result:
        LOGGER.debug("Received event: %s", event)

        if isinstance(event, ResponseOutputItemAddedEvent):
            if isinstance(event.item, ResponseOutputMessage):
                yield {"role": event.item.role}
            elif isinstance(event.item, ResponseFunctionToolCall):
                # OpenAI has tool calls as individual events
                # while HA puts tool calls inside the assistant message.
                # We turn them into individual assistant content for HA
                # to ensure that tools are called as soon as possible.
                yield {"role": "assistant"}
                current_tool_call = event.item
        elif isinstance(event, ResponseOutputItemDoneEvent):
            item = event.item.model_dump()
            item.pop("status", None)
            if isinstance(event.item, ResponseReasoningItem):
                messages.append(cast(ResponseReasoningItemParam, item))
            elif isinstance(event.item, ResponseOutputMessage):
                messages.append(cast(ResponseOutputMessageParam, item))
            elif isinstance(event.item, ResponseFunctionToolCall):
                messages.append(cast(ResponseFunctionToolCallParam, item))
        elif isinstance(event, ResponseTextDeltaEvent):
            yield {"content": event.delta}
        elif isinstance(event, ResponseFunctionCallArgumentsDeltaEvent):
            current_tool_call.arguments += event.delta
        elif isinstance(event, ResponseFunctionCallArgumentsDoneEvent):
            current_tool_call.status = "completed"
            yield {
                "tool_calls": [
                    llm.ToolInput(
                        id=current_tool_call.call_id,
                        tool_name=current_tool_call.name,
                        tool_args=json.loads(current_tool_call.arguments),
                    )
                ]
            }
        elif isinstance(event, ResponseCompletedEvent):
            if event.response.usage is not None:
                chat_log.async_trace(
                    {
                        "stats": {
                            "input_tokens": event.response.usage.input_tokens,
                            "output_tokens": event.response.usage.output_tokens,
                        }
                    }
                )
        elif isinstance(event, ResponseIncompleteEvent):
            if event.response.usage is not None:
                chat_log.async_trace(
                    {
                        "stats": {
                            "input_tokens": event.response.usage.input_tokens,
                            "output_tokens": event.response.usage.output_tokens,
                        }
                    }
                )

            if (
                    event.response.incomplete_details
                    and event.response.incomplete_details.reason
            ):
                reason: str = event.response.incomplete_details.reason
            else:
                reason = "unknown reason"

            if reason == "max_output_tokens":
                reason = "max output tokens reached"
            elif reason == "content_filter":
                reason = "content filter triggered"

            raise HomeAssistantError(f"OpenAI response incomplete: {reason}")
        elif isinstance(event, ResponseFailedEvent):
            if event.response.usage is not None:
                chat_log.async_trace(
                    {
                        "stats": {
                            "input_tokens": event.response.usage.input_tokens,
                            "output_tokens": event.response.usage.output_tokens,
                        }
                    }
                )
            reason = "unknown reason"
            if event.response.error is not None:
                reason = event.response.error.message
            raise HomeAssistantError(f"OpenAI response failed: {reason}")
        elif isinstance(event, ResponseErrorEvent):
            raise HomeAssistantError(f"OpenAI response error: {event.message}")


class LlamaConversationEntity(ConversationEntity, AbstractConversationAgent):
    """Llama Assist conversation agent."""

    _attr_has_entity_name = True
    _attr_name = None
    _attr_supports_streaming = True

    def __init__(self, entry: LlamaAPIClientsConfigEntry) -> None:
        """Initialize the agent."""
        self.entry = entry
        self._attr_name = entry.title
        self._attr_unique_id = entry.entry_id

        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name=entry.title,
            manufacturer="LlamaAssist",
            model="LocalLLM",
            entry_type=dr.DeviceEntryType.SERVICE,
        )

        settings = {**self.entry.data, **self.entry.options}
        apis = self.entry.runtime_data
        self.completion_client: AsyncOpenAI = apis.completion_client
        self.embeddings_client: AsyncOpenAI | None = apis.embeddings_client
        self.embeddings_db: EmbeddingsDatabase | None = apis.embeddings_db

        self._attr_supported_features = (
            ConversationEntityFeature.CONTROL
        )
        # TODO: investigate if we can add features here to access from llm context (see pipeline.py)

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return a list of supported languages."""
        return MATCH_ALL

    async def async_added_to_hass(self) -> None:
        """When entity is added to Home Assistant."""
        await super().async_added_to_hass()

        assist_pipeline.async_migrate_engine(
            self.hass, "conversation", self.entry.entry_id, self.entity_id
        )
        conversation.async_set_agent(self.hass, self.entry, self)
        self.entry.async_on_unload(
            self.entry.add_update_listener(self._async_entry_update_listener)
        )

    async def async_will_remove_from_hass(self) -> None:
        """When entity will be removed from Home Assistant."""
        conversation.async_unset_agent(self.hass, self.entry)
        await super().async_will_remove_from_hass()

    # async def async_process(self, user_input: ConversationInput) -> ConversationResult:
    #     """Process a sentence."""
    #     with (
    #         chat_session.async_get_chat_session(self.hass, user_input.conversation_id) as session,
    #         conversation.async_get_chat_log(self.hass, session, user_input) as chat_log,
    #     ):
    #         return await self._async_handle_message(user_input, chat_log)

    async def _async_handle_message(
            self,
            user_input: conversation.ConversationInput,
            chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        """Call the API."""
        settings = {**self.entry.data, **self.entry.options}

        try:
            await chat_log.async_update_llm_data(
                conversing_domain=DOMAIN,
                user_input=user_input,
                # user_llm_hass_api=settings.get(CONF_LLM_HASS_API),
                user_llm_hass_api=LLAMA_LLM_API,
                user_llm_prompt=settings.get(CONF_PROMPT),
            )
        except conversation.ConverseError as err:
            return err.as_conversation_result()

        tools_to_use: list[llm.Tool] = []
        if settings.get(CONF_USE_EMBEDDINGS_ENTITIES) or settings.get(CONF_USE_EMBEDDINGS_TOOLS):
            # If using embeddings, we need to get the user input vector
            user_input_vector = await self.embeddings_client.embeddings.create(input=user_input.text, model="none",
                                                                               encoding_format="float")

            if user_input_vector:
                user_input_vector = user_input_vector[0].model_extra.get('embedding')[0]

                if settings.get(CONF_USE_EMBEDDINGS_TOOLS):
                    await self.embeddings_db.store_tools(chat_log.llm_api.tools)
                    tools_to_use = await self.embeddings_db.matching_tools(user_input=user_input_vector)

                if settings.get(CONF_USE_EMBEDDINGS_ENTITIES):
                    # If using embeddings entities, we need to get the relevant entities and add them to the chat log for the LLM
                    if isinstance(chat_log.llm_api.api, LlamaAssistAPI):
                        _all_exposed_entities = chat_log.llm_api.api.all_exposed_entities

                        await self.embeddings_db.store_entities(_all_exposed_entities)
                        matching_entities = await self.embeddings_db.matching_entities(user_input=user_input_vector)

                        prompt = []

                        if matching_entities:
                            prompt.append(
                                "Static Context Update:"
                            )
                            prompt.append(yaml_util.dump(list(matching_entities.values())))

                        chat_log.content.insert(
                            len(chat_log.content) - 1,
                            SystemContent(content="\n".join(prompt))
                        )
            else:
                LOGGER.warning("No user input vector found, not using embeddings tools or entities.")

        await self._async_handle_chat_log(chat_log, tools_to_use)

        intent_response = intent.IntentResponse(language=user_input.language)
        assert type(chat_log.content[-1]) is conversation.AssistantContent
        intent_response.async_set_speech(chat_log.content[-1].content or "")
        return conversation.ConversationResult(
            response=intent_response,
            conversation_id=chat_log.conversation_id,
            continue_conversation=chat_log.continue_conversation,
        )

    async def _async_handle_chat_log(
            self,
            chat_log: conversation.ChatLog,
            tools: list[llm.Tool],
    ) -> None:
        """Generate an answer for the chat log."""

        # Convert tools to FunctionToolParam format
        tools = [
            _format_tool(tool, chat_log.llm_api.custom_serializer)
            for tool in tools
        ]

        # Convert chat log content to OpenAI message format
        messages = [
            m
            for content in chat_log.content
            for m in _convert_content_to_param(content)
        ]

        # To prevent infinite loops, we limit the number of iterations
        for _iteration in range(MAX_TOOL_ITERATIONS):
            model_args = {
                "model": "none",
                "input": messages,
                # "max_output_tokens": options.get(CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS),
                # "top_p": options.get(CONF_TOP_P, RECOMMENDED_TOP_P),
                # "temperature": options.get(CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE),
                "user": chat_log.conversation_id,
                "stream": True,
            }
            if tools:
                model_args["tools"] = tools

            # if model.startswith("o"):
            #     model_args["reasoning"] = {
            #         "effort": options.get(
            #             CONF_REASONING_EFFORT, RECOMMENDED_REASONING_EFFORT
            #         )
            #     }
            # else:
            #     model_args["store"] = False
            model_args["store"] = False

            try:
                result = await self.completion_client.responses.create(**model_args)
                result = await self.completion_client.chat.completions.create(**model_args)
            except openai.RateLimitError as err:
                LOGGER.error("Rate limited by OpenAI: %s", err)
                raise HomeAssistantError("Rate limited or insufficient funds") from err
            except openai.OpenAIError as err:
                LOGGER.error("Error talking to OpenAI: %s", err)
                raise HomeAssistantError("Error talking to OpenAI") from err

            async for content in chat_log.async_add_delta_content_stream(
                    self.entity_id, _transform_stream(chat_log, result, messages)
            ):
                if not isinstance(content, conversation.AssistantContent):
                    messages.extend(_convert_content_to_param(content))

            if not chat_log.unresponded_tool_results:
                break

    # def _trim_history(self, message_history: MessageHistory, max_messages: int) -> None:
    #     """Trims excess messages from a single history.
    #
    #     This sets the max history to allow a configurable size history may take
    #     up in the context window.
    #
    #     Note that some messages in the history may not be from ollama only, and
    #     may come from other anents, so the assumptions here may not strictly hold,
    #     but generally should be effective.
    #     """
    #     if max_messages < 1:
    #         # Keep all messages
    #         return
    #
    #     # Ignore the in progress user message
    #     num_previous_rounds = message_history.num_user_messages - 1
    #     if num_previous_rounds >= max_messages:
    #         # Trim history but keep system prompt (first message).
    #         # Every other message should be an assistant message, so keep 2x
    #         # message objects. Also keep the last in progress user message
    #         num_keep = 2 * max_messages + 1
    #         drop_index = len(message_history.messages) - num_keep
    #         message_history.messages = [message_history.messages[0]] + message_history.messages[drop_index:]

    async def _async_entry_update_listener(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        """Handle options update."""
        # Reload as we update device info + entity name + supported features
        await hass.config_entries.async_reload(entry.entry_id)

# def _parse_reasoning_message(msg: str) -> str:
#     msg = msg.replace("\n\n", "")
#
#     if "<think>" in msg and "</think>" in msg:
#         # Extract the thought content
#         return msg.split("</think>")[1].strip()
#     return msg
