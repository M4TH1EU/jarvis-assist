from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Optional, AsyncGenerator, Union

import httpx
from homeassistant.core import HomeAssistant
from homeassistant.helpers.httpx_client import get_async_client


# Exceptions
class RequestError(Exception):
    """Raised when an API request fails (e.g. network or client error)."""
    pass


class ResponseError(Exception):
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
    def __init__(self, hass: HomeAssistant, base_url: str = "http://localhost:8080"):
        self.base_url = base_url.rstrip("/")
        self._client = get_async_client(hass)

    async def chat(
            self,
            messages: Optional[list[Message]] = None,
            *,
            tools: Optional[list[Tool]] = None,
            stream: bool = False,
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

        if tools:
            payload["tools"] = [tool.to_dict() for tool in tools]

        try:
            if not stream:
                resp = await self._client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()
                choice = data["choices"][0]["message"]

                # build a one‐shot async generator
                async def fake_stream() -> AsyncGenerator[Message, None]:
                    tool_calls = [
                        ToolCall(
                            name=call["function"]["name"],
                            arguments=json.loads(call["function"]["arguments"])
                            # TODO: make sure JSON is valid and fix common issues with LLMs

                        )
                        for call in choice.get("tool_calls", [])
                    ]
                    # yield one message with both content and tool_calls, marked done
                    yield Message(
                        role=choice["role"],
                        content=choice.get("content").replace("\n\n", "", 1),  # remove the first \n\n
                        reasoning_content=choice.get("reasoning_content"),
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
