import json
import logging
import time
from typing import Any, AsyncGenerator, Dict, Iterable, List, Optional

from litellm.integrations.custom_logger import CustomLogger
from litellm.types.utils import ModelResponseStream

from opencode_compat_hook.parser import (
    find_raw_tool_start,
    has_any_dsml_prefix,
    has_complete_raw_tool_block,
    normalize_raw_tool_calls,
    parse_raw_tool_calls,
)


log = logging.getLogger("opencode_compat_hook")

SECTION_SIZE = 32
GUARD_SECTIONS = 2


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _set(obj: Any, key: str, value: Any) -> None:
    if isinstance(obj, dict):
        obj[key] = value
    else:
        setattr(obj, key, value)


def _choice(response: Any, index: int = 0) -> Any:
    choices = _get(response, "choices", []) or []
    if not choices or len(choices) <= index:
        return None
    return choices[index]


def _message(choice: Any) -> Any:
    return _get(choice, "message", {}) or {}


def _delta(chunk: Any) -> Any:
    choice = _choice(chunk)
    if choice is None:
        return {}
    return _get(choice, "delta", {}) or {}


def _content_from_delta(delta: Any) -> str:
    return _get(delta, "content", "") or ""


def _reasoning_from_delta(delta: Any) -> str:
    return _get(delta, "reasoning", "") or _get(delta, "reasoning_content", "") or ""


def _chunk_id(chunk: Any, fallback: str = "chatcmpl-opencode-compat") -> str:
    return _get(chunk, "id", None) or fallback


def _chunk_model(chunk: Any, fallback: str = "unknown") -> str:
    return _get(chunk, "model", None) or fallback


def _chunk_created(chunk: Any, fallback: Optional[int] = None) -> int:
    return _get(chunk, "created", None) or fallback or int(time.time())


def _make_stream_chunk(
    chunk_id: str,
    model: str,
    created: int,
    delta: Dict[str, Any],
    finish_reason: Optional[str] = None,
) -> ModelResponseStream:
    return ModelResponseStream(
        id=chunk_id,
        object="chat.completion.chunk",
        created=created,
        model=model,
        choices=[{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    )


def _make_content_chunk(chunk_id: str, model: str, created: int, text: str) -> ModelResponseStream:
    return _make_stream_chunk(chunk_id, model, created, {"content": text})


def build_stream_tool_call_chunks(
    tool_calls: Iterable[Dict[str, Any]], chunk_id: str, model: str, created: int
) -> List[ModelResponseStream]:
    chunks: List[ModelResponseStream] = []
    for tc in tool_calls:
        fn = tc["function"]
        chunks.append(
            _make_stream_chunk(
                chunk_id,
                model,
                created,
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": tc["id"],
                            "type": "function",
                            "function": {"name": fn["name"], "arguments": ""},
                        }
                    ]
                },
            )
        )
        args = fn["arguments"]
        for i in range(0, len(args), SECTION_SIZE):
            chunks.append(
                _make_stream_chunk(
                    chunk_id,
                    model,
                    created,
                    {"tool_calls": [{"index": 0, "function": {"arguments": args[i:i + SECTION_SIZE]}}]},
                )
            )

    chunks.append(_make_stream_chunk(chunk_id, model, created, {"content": ""}, finish_reason="tool_calls"))
    return chunks


def _request_url(request_data: Optional[dict]) -> str:
    if not request_data:
        return ""
    proxy_request = request_data.get("proxy_server_request") or {}
    if isinstance(proxy_request, dict):
        return str(proxy_request.get("url") or "")
    return ""


def _should_skip_stream_conversion(request_data: Optional[dict]) -> bool:
    if not request_data:
        return False

    call_type = str(request_data.get("call_type") or "")
    if call_type in {"anthropic_messages", "pass_through_endpoint"}:
        return True

    url = _request_url(request_data)
    if "/v1/messages" in url or "/messages" in url:
        return True

    return False


def convert_non_streaming_response(response: Any) -> Any:
    choice = _choice(response)
    if choice is None:
        return response

    msg = _message(choice)
    content = _get(msg, "content", "") or ""
    tool_calls = _get(msg, "tool_calls", None)

    if tool_calls or not content or not has_complete_raw_tool_block(content):
        return response

    parsed = parse_raw_tool_calls(normalize_raw_tool_calls(content))
    if not parsed:
        log.warning("raw tool block detected but parse returned empty: %s", content[:600])
        return response

    _set(msg, "tool_calls", parsed)
    _set(msg, "content", None)
    _set(choice, "finish_reason", "tool_calls")
    log.info("converted %d non-stream tool_calls", len(parsed))
    return response


class OpencodeCompatHandler(CustomLogger):
    """Compatibility layer for opencode raw DSML/Qwen tool-call output."""

    def __init__(self) -> None:
        self._register_input_tokens_route()

    def _register_input_tokens_route(self) -> None:
        try:
            from fastapi import Request
            from fastapi.responses import JSONResponse
            from litellm.proxy.proxy_server import app

            route_path = "/v1/responses/input_tokens"
            for route in getattr(app, "routes", []):
                if getattr(route, "path", None) == route_path:
                    return

            @app.post(route_path)
            async def opencode_responses_input_tokens(request: Request):
                body = await request.body()
                try:
                    payload = json.loads(body)
                    total_chars = 0
                    for item in payload.get("input", []) or []:
                        content = item.get("content", "") if isinstance(item, dict) else ""
                        if isinstance(content, str):
                            total_chars += len(content)
                        elif isinstance(content, list):
                            for part in content:
                                if isinstance(part, dict) and part.get("type") == "text":
                                    total_chars += len(part.get("text", ""))
                    tokens = max(1, total_chars // 4)
                except Exception:
                    tokens = 100
                return JSONResponse(content={"object": "response.input_tokens", "input_tokens": tokens})

            log.info("registered opencode compatibility route %s", route_path)
        except Exception as exc:
            log.warning("failed to register /v1/responses/input_tokens route: %s", exc)

    async def async_pre_call_hook(self, user_api_key_dict: Any, cache: Any, data: dict, call_type: str):
        if call_type not in ("completion", "acompletion", "chat_completion"):
            return data

        tools = data.get("tools")
        if isinstance(tools, list):
            data["tools"] = [tool for tool in tools if isinstance(tool, dict) and tool.get("type") == "function"]
            if not data["tools"]:
                data.pop("tools", None)

        for msg in data.get("messages", []) or []:
            if not isinstance(msg, dict):
                continue
            if msg.get("role") == "assistant" and not msg.get("content") and not msg.get("tool_calls"):
                msg["content"] = " "
        return data

    async def async_post_call_success_hook(self, data: dict, user_api_key_dict: Any, response: Any) -> Any:
        return convert_non_streaming_response(response)

    async def async_post_call_streaming_iterator_hook(
        self, user_api_key_dict: Any, response: Any, request_data: dict
    ) -> AsyncGenerator[Any, None]:
        if _should_skip_stream_conversion(request_data):
            async for chunk in response:
                yield chunk
            return

        buffer = ""
        unflushed_text = ""
        pending: List[str] = []
        dsml_mode = False
        content_collected = False
        thinking_state = 0
        last_id = "chatcmpl-opencode-compat"
        last_model = request_data.get("model", "unknown") if request_data else "unknown"
        last_created = int(time.time())

        async for chunk in response:
            # Native passthrough streams can be bytes; leave them untouched.
            if isinstance(chunk, (bytes, bytearray)):
                yield chunk
                continue

            last_id = _chunk_id(chunk, last_id)
            last_model = _chunk_model(chunk, last_model)
            last_created = _chunk_created(chunk, last_created)

            delta = _delta(chunk)
            if _get(delta, "role", None):
                yield chunk
                continue

            reasoning = _reasoning_from_delta(delta)
            text = _content_from_delta(delta)
            raw_chunk_text = reasoning or text

            if not raw_chunk_text:
                if not dsml_mode:
                    yield chunk
                continue

            is_reasoning = bool(reasoning)
            chunk_text = ""
            if is_reasoning:
                if thinking_state == 0:
                    chunk_text += "<think>\n"
                    thinking_state = 1
                chunk_text += raw_chunk_text
            else:
                if thinking_state == 1:
                    chunk_text += "\n</think>\n"
                    thinking_state = 2
                chunk_text += raw_chunk_text

            buffer += chunk_text
            content_collected = True

            if has_complete_raw_tool_block(buffer):
                idx = find_raw_tool_start(buffer)
                if idx > 0:
                    prefix = buffer[:idx]
                    if thinking_state == 1:
                        prefix += "\n</think>\n"
                        thinking_state = 2
                    yield _make_content_chunk(last_id, last_model, last_created, prefix)

                parsed = parse_raw_tool_calls(normalize_raw_tool_calls(buffer))
                if parsed:
                    log.info("converted %d streaming tool_calls", len(parsed))
                    for out_chunk in build_stream_tool_call_chunks(parsed, last_id, last_model, last_created):
                        yield out_chunk
                else:
                    log.warning("stream raw tool block detected but parse returned empty")
                    yield _make_content_chunk(last_id, last_model, last_created, buffer[idx:])
                    yield _make_stream_chunk(last_id, last_model, last_created, {"content": ""}, finish_reason="stop")
                return

            if dsml_mode:
                continue

            if has_any_dsml_prefix(buffer):
                dsml_mode = True
                for item in pending:
                    yield _make_content_chunk(last_id, last_model, last_created, item)
                pending.clear()
                if unflushed_text:
                    yield _make_content_chunk(last_id, last_model, last_created, unflushed_text)
                    unflushed_text = ""
                continue

            unflushed_text += chunk_text
            while len(unflushed_text) >= SECTION_SIZE:
                pending.append(unflushed_text[:SECTION_SIZE])
                unflushed_text = unflushed_text[SECTION_SIZE:]
                if len(pending) > GUARD_SECTIONS:
                    yield _make_content_chunk(last_id, last_model, last_created, pending.pop(0))

        if thinking_state == 1:
            unflushed_text += "\n</think>\n"

        if dsml_mode:
            if content_collected:
                idx = find_raw_tool_start(buffer)
                if idx > 0:
                    yield _make_content_chunk(last_id, last_model, last_created, buffer[:idx])
                if not has_complete_raw_tool_block(buffer) and idx < len(buffer):
                    yield _make_content_chunk(last_id, last_model, last_created, buffer[idx:])
        else:
            for item in pending:
                yield _make_content_chunk(last_id, last_model, last_created, item)
            if unflushed_text:
                yield _make_content_chunk(last_id, last_model, last_created, unflushed_text)

        yield _make_stream_chunk(last_id, last_model, last_created, {"content": ""}, finish_reason="stop")


proxy_handler_instance = OpencodeCompatHandler()
