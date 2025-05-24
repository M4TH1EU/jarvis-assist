import json
from typing import Literal, Callable, Any, AsyncGenerator

from homeassistant.components import assist_pipeline, conversation
from homeassistant.components.conversation import AbstractConversationAgent, ConversationEntityFeature, \
    ConversationInput, ConversationResult, AssistantContent
from homeassistant.components.conversation import ConversationEntity
from homeassistant.components.ollama.models import MessageRole
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import MATCH_ALL, CONF_LLM_HASS_API
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import llm, chat_session
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.intent import IntentResponse
from voluptuous_openapi import convert

from . import LOGGER, DOMAIN
from .const import CONF_PROMPT, CONF_MAX_HISTORY, DEFAULT_MAX_HISTORY
from .llamacpp_adapter import Message, MessageHistory

# Max number of back and forth with the LLM to generate a response
MAX_TOOL_ITERATIONS = 10


async def async_setup_entry(hass: HomeAssistant, config_entry: ConfigEntry,
                            async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    """Set up conversation entities."""
    agent = JarvisConversationEntity(config_entry)
    async_add_entities([agent])


def _format_tool(
        tool: llm.Tool, custom_serializer: Callable[[Any], Any] | None
) -> dict[str, Any]:
    """Format tool specification."""
    tool_spec = {
        "name": tool.name,
        "parameters": convert(tool.parameters, custom_serializer=custom_serializer),
    }
    if tool.description:
        tool_spec["description"] = tool.description
    return {"type": "function", "function": tool_spec}


def _fix_invalid_arguments(value: Any) -> Any:
    """Attempt to repair incorrectly formatted json function arguments.

    Small models (for example llama3.1 8B) may produce invalid argument values
    which we attempt to repair here.
    """
    if not isinstance(value, str):
        return value
    if (value.startswith("[") and value.endswith("]")) or (
            value.startswith("{") and value.endswith("}")
    ):
        try:
            return json.loads(value)
        except json.decoder.JSONDecodeError:
            pass
    return value


def _parse_tool_args(arguments: dict[str, Any]) -> dict[str, Any]:
    """Rewrite ollama tool arguments.

    This function improves tool use quality by fixing common mistakes made by
    small local tool use models. This will repair invalid json arguments and
    omit unnecessary arguments with empty values that will fail intent parsing.
    """
    return {k: _fix_invalid_arguments(v) for k, v in arguments.items() if v}


def _convert_content(
        chat_content: (
                conversation.Content
                | conversation.ToolResultContent
                | conversation.AssistantContent
        ),
) -> Message:
    """Create tool response content."""
    if isinstance(chat_content, conversation.ToolResultContent):
        return Message(
            role=MessageRole.TOOL,
            content=json.dumps(chat_content.tool_result),
        )
    if isinstance(chat_content, conversation.AssistantContent):
        return Message(
            role=MessageRole.ASSISTANT,
            content=chat_content.content,
            tool_calls=[
                Message.ToolCall(
                    function=Message.ToolCall.Function(
                        name=tool_call.tool_name,
                        arguments=tool_call.tool_args,
                    )
                )
                for tool_call in chat_content.tool_calls or ()
            ],
        )
    if isinstance(chat_content, conversation.UserContent):
        return Message(
            role=MessageRole.USER,
            content=chat_content.content,
        )
    if isinstance(chat_content, conversation.SystemContent):
        return Message(
            role=MessageRole.SYSTEM,
            content=chat_content.content,
        )
    raise TypeError(f"Unexpected content type: {type(chat_content)}")


async def _transform_stream(
        result: AsyncGenerator[Message],
) -> AsyncGenerator[conversation.AssistantContentDeltaDict]:
    """Transform the response stream into HA format.

    An Ollama streaming response may come in chunks like this:

    response: message=Message(role="assistant", content="Paris")
    response: message=Message(role="assistant", content=".")
    response: message=Message(role="assistant", content=""), done: True, done_reason: "stop"
    response: message=Message(role="assistant", tool_calls=[...])
    response: message=Message(role="assistant", content=""), done: True, done_reason: "stop"

    This generator conforms to the chatlog delta stream expectations in that it
    yields deltas, then the role only once the response is done.
    """

    new_msg = True
    async for response in result:
        LOGGER.debug("Received response: %s", response)
        response_message = response["message"]
        chunk: conversation.AssistantContentDeltaDict = {}
        if new_msg:
            new_msg = False
            chunk["role"] = "assistant"
        if (tool_calls := response_message.get("tool_calls")) is not None:
            chunk["tool_calls"] = [
                llm.ToolInput(
                    tool_name=tool_call["function"]["name"],
                    tool_args=_parse_tool_args(tool_call["function"]["arguments"]),
                )
                for tool_call in tool_calls
            ]
        if (content := response_message.get("content")) is not None:
            chunk["content"] = content
        if response_message.get("done"):
            new_msg = True
        yield chunk


class JarvisConversationEntity(ConversationEntity, AbstractConversationAgent):
    """Jarvis Assist conversation agent."""

    _attr_has_entity_name = True
    _attr_supports_streaming = True

    def __init__(self, entry: ConfigEntry) -> None:
        """Initialize the agent."""
        self.entry = entry

        # conversation id -> message history
        self._attr_name = entry.title
        self._attr_unique_id = entry.entry_id

        # if self.entry.options.get(DOMAIN):
        self._attr_supported_features = (
            ConversationEntityFeature.CONTROL
        )

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

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return a list of supported languages."""
        return MATCH_ALL

    async def async_process(self, user_input: ConversationInput) -> ConversationResult:
        """Process a sentence."""
        with (
            chat_session.async_get_chat_session(self.hass, user_input.conversation_id) as session,
            conversation.async_get_chat_log(self.hass, session, user_input) as chat_log,
        ):
            return await self._async_handle_message(user_input, chat_log)

    async def _async_handle_message(
            self,
            user_input: conversation.ConversationInput,
            chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        """Call the API."""
        settings = {**self.entry.data, **self.entry.options}

        use_stream = False

        client = self.hass.data[DOMAIN][self.entry.entry_id]

        try:
            await chat_log.async_update_llm_data(
                DOMAIN,
                user_input,
                settings.get(CONF_LLM_HASS_API),
                settings.get(CONF_PROMPT),
            )
        except conversation.ConverseError as err:
            return err.as_conversation_result()

        tools: list[dict[str, Any]] | None = None
        if chat_log.llm_api:
            tools = [
                _format_tool(tool, chat_log.llm_api.custom_serializer)
                for tool in chat_log.llm_api.tools
            ]

        message_history: MessageHistory = MessageHistory(
            [_convert_content(content) for content in chat_log.content]
        )
        max_messages = int(settings.get(CONF_MAX_HISTORY, DEFAULT_MAX_HISTORY))
        self._trim_history(message_history, max_messages)

        # Get response
        # To prevent infinite loops, we limit the number of iterations
        for _iteration in range(MAX_TOOL_ITERATIONS):
            try:
                response_generator = await client.chat(
                    # Make a copy of the messages because we mutate the list later
                    messages=list(message_history.messages),
                    tools=tools,
                    stream=use_stream
                )
            # except (ollama.RequestError, ollama.ResponseError) as err:
            #     _LOGGER.error("Unexpected error talking to Ollama server: %s", err)
            #     raise HomeAssistantError(
            #         f"Sorry, I had a problem talking to the Ollama server: {err}"
            #     ) from err
            except Exception as err:
                LOGGER.error("Unexpected error talking to llama.cpp server: %s", err)
                raise HomeAssistantError(f"Sorry, I had a problem talking to the llama.cpp server: {err}") from err

            if use_stream:
                message_history.messages.extend(
                    _convert_content(content)
                    async for content in chat_log.async_add_delta_content_stream(
                        user_input.agent_id,
                        _transform_stream(response_generator),
                    )
                )
            else:
                chat_log.content.append(
                    AssistantContent(
                        content=_parse_reasoning_message(response_generator.message.content or ""),
                        agent_id=user_input.agent_id,
                        tool_calls=[
                            llm.ToolInput(
                                tool_name=tc.function.name,
                                tool_args=tc.function.arguments,
                            )
                            for tc in response_generator.message.tool_calls
                        ],
                    )
                )

                message_history.messages.append(
                    Message(
                        role=response_generator.message.role,
                        content=response_generator.message.content,
                        tool_calls=response_generator.message.tool_calls,
                        done=response_generator.message.done,
                        done_reason=response_generator.message.done_reason,
                    )
                )

            if not chat_log.unresponded_tool_results:
                break

        # Create intent response
        intent_response = IntentResponse(language=user_input.language)
        if not isinstance(chat_log.content[-1], conversation.AssistantContent):
            raise TypeError(
                f"Unexpected last message type: {type(chat_log.content[-1])}"
            )
        intent_response.async_set_speech(chat_log.content[-1].content or "")
        return conversation.ConversationResult(
            response=intent_response,
            conversation_id=chat_log.conversation_id,
            continue_conversation=chat_log.continue_conversation or False,
        )

    def _trim_history(self, message_history: MessageHistory, max_messages: int) -> None:
        """Trims excess messages from a single history.

        This sets the max history to allow a configurable size history may take
        up in the context window.

        Note that some messages in the history may not be from ollama only, and
        may come from other anents, so the assumptions here may not strictly hold,
        but generally should be effective.
        """
        if max_messages < 1:
            # Keep all messages
            return

        # Ignore the in progress user message
        num_previous_rounds = message_history.num_user_messages - 1
        if num_previous_rounds >= max_messages:
            # Trim history but keep system prompt (first message).
            # Every other message should be an assistant message, so keep 2x
            # message objects. Also keep the last in progress user message
            num_keep = 2 * max_messages + 1
            drop_index = len(message_history.messages) - num_keep
            message_history.messages = [message_history.messages[0]] + message_history.messages[drop_index:]

    async def _async_entry_update_listener(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        """Handle options update."""
        # Reload as we update device info + entity name + supported features
        await hass.config_entries.async_reload(entry.entry_id)


def _parse_reasoning_message(msg: str) -> str:
    if "<think>" in msg and "</think>" in msg:
        # Extract the thought content
        return msg.split("</think>")[1].strip()
    return msg
