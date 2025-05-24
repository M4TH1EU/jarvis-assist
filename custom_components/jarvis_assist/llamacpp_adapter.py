from __future__ import annotations

import json
from collections.abc import AsyncGenerator as _AsyncGen
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Callable, Literal, Mapping, Optional, Sequence, Union

import httpx


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
    """Chat message history."""

    messages: list[Message]
    """List of message history, including system prompt and assistant responses."""

    @property
    def num_user_messages(self) -> int:
        """Return a count of user messages."""
        return sum(m["role"] == MessageRole.USER.value for m in self.messages)


# Base model to allow dict-like access
class SubscriptableBaseModel:
    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key) if hasattr(self, key) else default


# Structures returned by the API
class BaseGenerateResponse(SubscriptableBaseModel):
    def __init__(
            self,
            model: Optional[str] = None,
            created_at: Optional[str] = None,
            done: Optional[bool] = None,
            done_reason: Optional[str] = None,
            total_duration: Optional[int] = None,
            load_duration: Optional[int] = None,
            prompt_eval_count: Optional[int] = None,
            prompt_eval_duration: Optional[int] = None,
            eval_count: Optional[int] = None,
            eval_duration: Optional[int] = None,
    ):
        self.model = model
        self.created_at = created_at
        self.done = done
        self.done_reason = done_reason
        self.total_duration = total_duration
        self.load_duration = load_duration
        self.prompt_eval_count = prompt_eval_count
        self.prompt_eval_duration = prompt_eval_duration
        self.eval_count = eval_count
        self.eval_duration = eval_duration


class Message(SubscriptableBaseModel):
    class ToolCall(SubscriptableBaseModel):
        class Function(SubscriptableBaseModel):
            def __init__(self, name: str, arguments: dict[str, Any]):
                self.name = name
                self.arguments = arguments

        def __init__(self, function: Message.ToolCall.Function):
            self.function = function

    def __init__(
            self,
            role: str,
            content: Optional[str] = None,
            tool_calls: Optional[list[Message.ToolCall]] = None,
            done: Optional[bool] = None,
            done_reason: Optional[str] = None,
    ):
        self.role = role
        self.content = content
        self.tool_calls = tool_calls or []
        self.done = done
        self.done_reason = done_reason


class ChatResponse(BaseGenerateResponse):
    def __init__(
            self,
            message: Message,
            **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.message = message


# Tool and Options types
class Tool(SubscriptableBaseModel):
    def __init__(
            self,
            name: str,
            parameters: Mapping[str, Any],
            description: Optional[str] = None,
    ):
        self.name = name
        self.parameters = parameters
        self.description = description


class Options(SubscriptableBaseModel):
    def __init__(self, **options: Any):
        for k, v in options.items():
            setattr(self, k, v)


# The client that makes API calls
class LlamaCppClient:
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(base_url=self.base_url)

    async def chat(
            self,
            model: str = "",
            messages: Optional[Sequence[Union[Mapping[str, Any], Message]]] = None,
            *,
            tools: Optional[Sequence[Union[Mapping[str, Any], Tool, Callable]]] = None,
            stream: Literal[False] = False,
            options: Optional[Union[Mapping[str, Any], Options]] = None,
            keep_alive: Optional[Union[float, str]] = None,
    ) -> Union[ChatResponse, _AsyncGen[Message]]:
        """
        Call the /v1/chat/completions endpoint on the Llama.cpp OpenAI-compatible server.
        Returns a ChatResponse if stream=False, otherwise an async generator yielding Message.
        """
        url = "/v1/chat/completions"
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                m if isinstance(m, Mapping) else m.__dict__ for m in (messages or [])
            ],
            "stream": stream,
        }
        if tools:
            payload["tools"] = tools

        if options:
            payload["options"] = (
                options if isinstance(options, Mapping) else options.__dict__
            )
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive

        try:
            if not stream:
                resp = await self._client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()
                choice = data["choices"][0]["message"]

                tool_calls: list[Message.ToolCall] = []
                for call in choice.get("tool_calls", []):
                    fn = call["function"]
                    args = fn.get("arguments", {})
                    # sometimes arguments come back as a JSON string
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            pass
                    tool_calls.append(
                        Message.ToolCall(
                            Message.ToolCall.Function(
                                name=fn["name"],
                                arguments=args,
                            )
                        )
                    )

                msg = Message(
                    role=choice["role"],
                    content=choice.get("content"),
                    tool_calls=tool_calls,
                    done=True,
                    done_reason=data["choices"][0].get("finish_reason"),
                )

                return ChatResponse(
                    message=msg,
                    model=data.get("model"),
                    created_at=data.get("created"),
                    done=data["choices"][0].get("finish_reason") is not None,
                    done_reason=data["choices"][0].get("finish_reason"),
                )
            else:
                # async def _stream() -> AsyncGenerator[Message, None]:
                #     async with self._client.stream("POST", url, json=payload) as resp:
                #         resp.raise_for_status()
                #         async for line in resp.aiter_lines():
                #             if not line or line == "[DONE]":
                #                 break
                #             # Lines are prefixed "data: "
                #             if line.startswith("data:"):
                #                 raw = line[len("data:"):].strip()
                #                 part = json.loads(raw)
                #                 # Each part has choices array
                #                 for choice in part.get("choices", []):
                #                     delta = choice.get("delta", {})
                #                     role = delta.get("role")
                #                     content = delta.get("content")
                #                     fc = delta.get("function_call")
                #                     tool_calls = []
                #                     if fc:
                #                         # accumulate arguments in streaming
                #                         tool_calls = [
                #                             Message.ToolCall(
                #                                 Message.ToolCall.Function(
                #                                     name=fc.get("name"),
                #                                     arguments=json.loads(fc.get("arguments", "{}"))
                #                                 )
                #                             )
                #                         ]
                #                     yield Message(
                #                         role=role,
                #                         content=content,
                #                         tool_calls=tool_calls,
                #                         done=choice.get("finish_reason") is not None,
                #                         done_reason=choice.get("finish_reason"),
                #                     )
                #
                # return _stream()
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