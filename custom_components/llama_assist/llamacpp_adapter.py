from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Optional, AsyncGenerator, Union

import httpx
from homeassistant.core import HomeAssistant
from homeassistant.helpers.httpx_client import get_async_client

from .const import LOGGER


# Exceptions
class RequestError(BaseException):
    """Raised when an API request fails (e.g. network or client error)."""
    pass


class ResponseError(BaseException):
    """Raised when the API returns an error status or payload."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


class MessageRole(StrEnum):
    """Role of a chat message."""

    SYSTEM = "system"  # prompt
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class MessageHistory:
    messages: list[Message]

    @property
    def num_user_messages(self) -> int:
        """Return a count of user messages."""
        return sum(m.role == MessageRole.USER for m in self.messages)


@dataclass
class Message:
    role: Union[MessageRole, str]
    content: Optional[str] = None
    reasoning_content: Optional[str] = None
    tool_calls: Optional[list[ToolCall]] = None
    done: Optional[bool] = None
    done_reason: Optional[str] = None

    def to_dict(self):
        """Convert the message to a dictionary for API requests."""
        data = {
            "role": self.role,
            "content": self.content,
            "done": self.done,
            "done_reason": self.done_reason,
        }
        if self.tool_calls:
            data["tool_calls"] = [
                {"type": "function", "function": {"name": call.name, "arguments": json.dumps(call.arguments)}}
                for call in self.tool_calls
            ]
        return data


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass
class Tool:
    type = "function"

    @dataclass
    class Function:
        """Function tool definition."""
        name: str
        parameters: dict
        description: Optional[str] = None

    function: Function

    def to_dict(self) -> dict[str, Any]:
        """Convert the tool to a dictionary for API requests."""
        return {
            "type": self.type,
            "function": {
                "name": self.function.name,
                "parameters": self.function.parameters,
                "description": self.function.description,
            }
        }


# The client that makes API calls
class LlamaCppClient:
    def __init__(self, hass: HomeAssistant, base_url: str = "", embeddings_base_url: str = ""):
        self.base_url = base_url.rstrip("/")
        self.embeddings_base_url = embeddings_base_url.rstrip("/") if embeddings_base_url else ""
        self._client = get_async_client(hass)

    async def chat(
            self,
            messages: Optional[list[Message]] = None,
            *,
            tools: Optional[list[Tool]] = None,
            stream: bool = False,
            disable_reasoning: bool = False
    ) -> AsyncGenerator[Message, None]:
        """
        Call the /v1/chat/completions endpoint on the Llama.cpp OpenAI-compatible server.
        Returns a ChatResponse if stream=False, otherwise an async generator yielding Message.
        """
        url = self.base_url + "/v1/chat/completions"
        payload: dict[str, Any] = {
            "messages": [msg.to_dict() for msg in messages] if messages else [],
            "stream": stream,
        }

        # If reasoning is disabled, append /no_think to the first user message content
        if disable_reasoning:
            id_of_last_msg_from_user = next(
                (i for i, msg in enumerate(payload["messages"]) if msg["role"] == MessageRole.USER),
                None
            )
            if id_of_last_msg_from_user is not None:
                payload["messages"][id_of_last_msg_from_user]["content"] += " /no_think"

        if tools:
            payload["tools"] = [tool.to_dict() for tool in tools]

        try:
            if not stream:
                resp = await self._client.post(url, json=payload, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                choice = data["choices"][0]["message"]

                content = choice.get("content").replace("\n\n", "", 1)
                reasoning_content = choice.get("reasoning_content", "")

                LOGGER.debug(f"""
                Payload Details:
                 - Tools: {', '.join(tool.function.name for tool in tools or [])}
                 - Latest system message: {payload["messages"][-2].get("content", "") if payload["messages"][-2]["role"] == MessageRole.SYSTEM else "<no system message>"}
                
                Llama.cpp Details:
                 - Time (prompt) : {data.get("timings").get("prompt_ms")} ms
                 - Time (completion) : {data.get("timings").get("predicted_ms")} ms
                 
                 - Tokens (prompt) : {data.get("usage").get("prompt_tokens")}
                 - Tokens (completion) : {data.get("usage").get("completion_tokens")}
                 
                 - Content : {content}
                 - Reasoning Content : {reasoning_content}
                 - Tool Calls : {choice.get("tool_calls", [])}
                 
                 Full payload: {json.dumps(payload)}
                 
                """)

                tool_calls = [
                    ToolCall(
                        name=call["function"]["name"],
                        arguments=json.loads(call["function"]["arguments"])
                        # TODO: make sure JSON is valid and fix common issues with LLMs
                    )
                    for call in choice.get("tool_calls", [])
                ]

                # Try to parse service call from content if the model didn't return a tool call
                if not tool_calls:
                    parsed_tools = parse_service_str(content, tools)
                    if parsed_tools:
                        tool_calls.extend(parsed_tools)
                        content = ""

                # build a one‐shot async generator
                async def fake_stream() -> AsyncGenerator[Message, None]:

                    # yield one message with both content and tool_calls, marked done
                    yield Message(
                        role=choice["role"],
                        content=content,  # remove the first \n\n
                        reasoning_content=reasoning_content,
                        tool_calls=tool_calls,
                        done=True,
                        done_reason="stop",
                    )

                return fake_stream()
            else:
                raise NotImplementedError("Streaming not supported yet")

        except httpx.HTTPStatusError as e:
            msg = e.response.text
            raise ResponseError(msg, e.response.status_code)
        except httpx.RequestError as e:
            raise RequestError(str(e))

    async def health(self) -> bool:
        """Ping the server to check if it's alive."""
        url = f"{self.base_url}/health"
        try:
            resp = await self._client.get(url)
            resp.raise_for_status()
            return resp.json().get("status") == "ok"
        except httpx.HTTPStatusError as e:
            raise ResponseError(e.response.text, e.response.status_code)
        except httpx.RequestError as e:
            raise RequestError(str(e))

    async def embeddings(self, texts: list[str]) -> list[list[float]]:
        """Call the /v1/embeddings endpoint to get embeddings for a list of texts."""
        if not self.embeddings_base_url:
            raise ValueError("Embeddings base URL is not set.")

        url = f"{self.embeddings_base_url}/v1/embeddings"
        payload = {
            "input": texts,
            "encoding_format": "float",
        }

        try:
            resp = await self._client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return [item["embedding"] for item in data["data"]]
        except httpx.HTTPStatusError as e:
            raise ResponseError(e.response.text, e.response.status_code)
        except httpx.RequestError as e:
            raise RequestError(str(e))


def parse_service_str(content: str, existing_tools: list[Tool]) -> Optional[list[ToolCall]]:
    """Try to parse the service call from the content."""

    lines = content.splitlines()
    first_line = lines[0] if lines else ""
    rest_of_lines = "\n".join(lines[1:]) if len(lines) > 1 else ""

    service_name = ""
    arguments = {}

    def _does_service_exist(service_name: str) -> bool:
        """Check if the service exists in the existing tools."""
        for tool in existing_tools:
            if tool.function.name == service_name:
                return True
        return False

    # Example Case 1:
    # HassListAddItem
    # item: cucumbers
    # name: Shopping List

    # Example Case 2:
    # HassListAddItem
    # { item: cucumbers, name: Shopping List }

    # Example Case 3:
    # HassListAddItem: { item: cucumbers, name: Shopping List }
    # TODO: implement

    # Example Case 4:
    # HassListAddItem: item: cucumbers, name: Shopping List
    # TODO: implement

    # Example Case 5:
    # HassListAddItem(item="cucumbers", name="Shopping List").
    # TODO: implement

    # Example Case 6:
    # HassListAddItem: item=cucumbers, name=Shopping List
    # TODO: implement

    try:
        if _does_service_exist(first_line):
            # The first line is the service name
            service_name = first_line

            # Case 2
            if "{" in rest_of_lines and "}" in rest_of_lines:
                # The rest of the lines are the arguments
                arguments = json.loads(rest_of_lines)

            # Case 1
            elif ":" in rest_of_lines:
                # The rest of the lines are key-value pairs
                arguments = {}
                for line in rest_of_lines.splitlines():
                    if ":" in line:
                        key, value = line.split(":", 1)
                        arguments[key.strip()] = value.strip()

        if service_name and arguments:
            return [
                ToolCall(
                    name=service_name,
                    arguments=arguments
                )
            ]
    except Exception:
        return None

    return None
