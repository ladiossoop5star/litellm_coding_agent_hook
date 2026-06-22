import json
import logging
import time
import uuid
from typing import Any, AsyncGenerator, Dict, Iterable, List, Optional, Tuple

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
ASSISTANT_PLACEHOLDER = "."
STOP_AFTER_FIRST_NATIVE_TOOL_MODEL_MARKER = "deepseek"


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


def _is_messages_stream(request_data: Optional[dict]) -> bool:
    if not request_data:
        return False
    call_type = str(request_data.get("call_type") or "")
    if call_type == "anthropic_messages":
        return True
    url = _request_url(request_data)
    return "/v1/messages" in url or "/messages" in url


def _should_skip_stream_conversion(request_data: Optional[dict]) -> bool:
    if not request_data:
        return False

    call_type = str(request_data.get("call_type") or "")
    if call_type == "pass_through_endpoint":
        return True

    return False


def _request_model_names(request_data: Optional[dict]) -> set[str]:
    if not request_data:
        return set()
    metadata = request_data.get("litellm_metadata") or {}
    values = {
        request_data.get("model"),
        metadata.get("model_group"),
        metadata.get("deployment"),
        metadata.get("deployment_model_name"),
    }
    return {str(value) for value in values if value}


def _stop_after_first_native_tool(request_data: Optional[dict]) -> bool:
    return any(STOP_AFTER_FIRST_NATIVE_TOOL_MODEL_MARKER in name.lower() for name in _request_model_names(request_data))


def convert_non_streaming_response(response: Any) -> Any:
    if isinstance(response, dict) and isinstance(response.get("content"), list):
        return _convert_anthropic_message_response(response)

    choice = _choice(response)
    if choice is None:
        return response

    msg = _message(choice)
    content = _get(msg, "content", "") or ""
    reasoning = _get(msg, "reasoning_content", "") or _get(msg, "reasoning", "") or ""
    tool_calls = _get(msg, "tool_calls", None)
    raw_text = content or reasoning

    if tool_calls or not raw_text or not has_complete_raw_tool_block(raw_text):
        return response

    parsed = parse_raw_tool_calls(normalize_raw_tool_calls(raw_text))
    if not parsed:
        log.warning("raw tool block detected but parse returned empty: %s", raw_text[:600])
        return response

    _set(msg, "tool_calls", parsed)
    _set(msg, "content", None)
    if reasoning:
        _set(msg, "reasoning_content", None)
    _set(choice, "finish_reason", "tool_calls")
    log.info("converted %d non-stream tool_calls", len(parsed))
    return response


def _tool_input(arguments: str) -> Dict[str, Any]:
    try:
        parsed = json.loads(arguments or "{}")
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _convert_anthropic_message_response(response: Dict[str, Any]) -> Dict[str, Any]:
    content = response.get("content") or []
    if not isinstance(content, list):
        return response

    for index, block in enumerate(content):
        if not isinstance(block, dict):
            continue
        text = block.get("thinking") if block.get("type") == "thinking" else block.get("text")
        if not isinstance(text, str) or not has_complete_raw_tool_block(text):
            continue

        parsed = parse_raw_tool_calls(normalize_raw_tool_calls(text))
        if not parsed:
            log.warning("anthropic raw tool block detected but parse returned empty: %s", text[:600])
            return response

        raw_start = find_raw_tool_start(text)
        prefix = text[:raw_start].rstrip()
        new_content: List[Dict[str, Any]] = []
        new_content.extend(content[:index])
        if prefix:
            new_block = dict(block)
            if new_block.get("type") == "thinking":
                new_block["thinking"] = prefix
            else:
                new_block["text"] = prefix
            new_content.append(new_block)

        for tc in parsed:
            fn = tc["function"]
            new_content.append(
                {
                    "type": "tool_use",
                    "id": tc.get("id") or "toolu_" + uuid.uuid4().hex[:24],
                    "name": fn["name"],
                    "input": _tool_input(fn.get("arguments") or "{}"),
                }
            )

        response["content"] = new_content
        response["stop_reason"] = "tool_use"
        log.info("converted %d anthropic non-stream tool_use blocks", len(parsed))
        return response

    return response


def _has_anthropic_text_or_tool_use(content: List[Any]) -> bool:
    for part in content:
        if not isinstance(part, dict):
            continue
        part_type = part.get("type")
        if part_type == "tool_use":
            return True
        if part_type == "text" and str(part.get("text") or "").strip():
            return True
    return False


def _normalize_assistant_messages(messages: Any) -> None:
    for msg in messages or []:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue

        content = msg.get("content")
        if isinstance(content, list):
            if not _has_anthropic_text_or_tool_use(content):
                content.append({"type": "text", "text": ASSISTANT_PLACEHOLDER})
            continue

        if not content and not msg.get("tool_calls"):
            msg["content"] = ASSISTANT_PLACEHOLDER


def _encode_like(text: str, original: Any) -> Any:
    if isinstance(original, (bytes, bytearray)):
        return text.encode("utf-8")
    return text


def _parse_sse_event(raw_event: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    event_name: Optional[str] = None
    data_lines: List[str] = []
    for line in raw_event.splitlines():
        if line.startswith("event:"):
            event_name = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:"):].strip())
    if not data_lines:
        return event_name, None
    data = "\n".join(data_lines)
    try:
        return event_name, json.loads(data)
    except Exception:
        return event_name, None


def _sse(event_name: str, payload: Dict[str, Any], original: Any) -> Any:
    text = "event: " + event_name + "\n"
    text += "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"
    return _encode_like(text, original)


def _is_complete_json_object(text: str) -> bool:
    try:
        return isinstance(json.loads(text), dict)
    except Exception:
        return False


def _messages_text_delta(text: str, index: int, original: Any, delta_type: str = "text_delta") -> Any:
    field = "thinking" if delta_type == "thinking_delta" else "text"
    return _sse(
        "content_block_delta",
        {"type": "content_block_delta", "index": index, "delta": {"type": delta_type, field: text}},
        original,
    )


def _raw_think_state() -> Dict[str, Any]:
    return {"in_think": False, "tail": ""}


def _matching_prefix_suffix(text: str, marker: str) -> str:
    max_len = min(len(text), len(marker) - 1)
    for size in range(max_len, 0, -1):
        if marker.startswith(text[-size:]):
            return text[-size:]
    return ""


def _strip_raw_think_delta(text: str, state: Dict[str, Any]) -> str:
    """Drop raw <think>...</think> text from model deltas, including split markers."""
    if not text:
        return ""

    open_marker = "<think>"
    close_marker = "</think>"
    data = str(state.get("tail") or "") + text
    state["tail"] = ""
    output: List[str] = []

    while data:
        if state.get("in_think"):
            close_idx = data.find(close_marker)
            if close_idx == -1:
                state["tail"] = _matching_prefix_suffix(data, close_marker)
                return "".join(output)
            data = data[close_idx + len(close_marker):]
            state["in_think"] = False
            continue

        open_idx = data.find(open_marker)
        if open_idx == -1:
            tail = _matching_prefix_suffix(data, open_marker)
            if tail:
                output.append(data[:-len(tail)])
                state["tail"] = tail
            else:
                output.append(data)
            return "".join(output)

        output.append(data[:open_idx])
        data = data[open_idx + len(open_marker):]
        state["in_think"] = True

    return "".join(output)


def _flush_raw_think_tail(state: Dict[str, Any]) -> str:
    tail = str(state.get("tail") or "")
    state["tail"] = ""
    return "" if state.get("in_think") else tail


def _messages_tool_use_events(tool_calls: Iterable[Dict[str, Any]], start_index: int, original: Any) -> List[Any]:
    events: List[Any] = []
    index = start_index
    for tc in tool_calls:
        fn = tc["function"]
        tool_id = "toolu_" + uuid.uuid4().hex[:24]
        args = fn.get("arguments") or "{}"
        events.append(
            _sse(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": index,
                    "content_block": {"type": "tool_use", "id": tool_id, "name": fn["name"], "input": {}},
                },
                original,
            )
        )
        events.append(
            _sse(
                "content_block_delta",
                {"type": "content_block_delta", "index": index, "delta": {"type": "input_json_delta", "partial_json": args}},
                original,
            )
        )
        events.append(_sse("content_block_stop", {"type": "content_block_stop", "index": index}, original))
        index += 1

    events.append(
        _sse(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use", "stop_sequence": None},
                "usage": {"output_tokens": 0},
            },
            original,
        )
    )
    events.append(_sse("message_stop", {"type": "message_stop"}, original))
    return events


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
        if call_type in ("completion", "acompletion", "chat_completion"):
            tools = data.get("tools")
            if isinstance(tools, list):
                data["tools"] = [tool for tool in tools if isinstance(tool, dict) and tool.get("type") == "function"]
                if not data["tools"]:
                    data.pop("tools", None)

        if call_type not in ("completion", "acompletion", "chat_completion", "anthropic_messages"):
            return data

        _normalize_assistant_messages(data.get("messages"))
        return data

    async def async_post_call_success_hook(self, data: dict, user_api_key_dict: Any, response: Any) -> Any:
        return convert_non_streaming_response(response)

    async def async_post_call_streaming_iterator_hook(
        self, user_api_key_dict: Any, response: Any, request_data: dict
    ) -> AsyncGenerator[Any, None]:
        if _is_messages_stream(request_data):
            async for chunk in self._convert_anthropic_messages_stream(
                response,
                stop_after_first_native_tool=_stop_after_first_native_tool(request_data),
            ):
                yield chunk
            return

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

    async def _convert_anthropic_messages_stream(
        self,
        response: Any,
        stop_after_first_native_tool: bool = False,
    ) -> AsyncGenerator[Any, None]:
        text_buffer = ""
        unflushed_text = ""
        pending: List[str] = []
        dsml_mode = False
        sse_buffer = ""
        text_block_index = 0
        text_delta_type = "text_delta"
        raw_think = _raw_think_state()
        native_tool_index: Optional[int] = None
        native_tool_json = ""
        passthrough_blocked = False
        original_for_output: Any = b""

        async for chunk in response:
            original_for_output = chunk
            chunk_text = chunk.decode("utf-8", errors="replace") if isinstance(chunk, (bytes, bytearray)) else str(chunk)
            sse_buffer += chunk_text

            while "\n\n" in sse_buffer:
                raw_event, sse_buffer = sse_buffer.split("\n\n", 1)
                if not raw_event:
                    continue

                event_name, payload = _parse_sse_event(raw_event)
                if payload is None:
                    if not passthrough_blocked:
                        yield _encode_like(raw_event + "\n\n", chunk)
                    continue

                if event_name == "content_block_delta":
                    delta = payload.get("delta") or {}
                    delta_type = str(delta.get("type") or "")
                    delta_field = "thinking" if delta_type == "thinking_delta" else "text"
                    if delta_type in {"text_delta", "thinking_delta"} and isinstance(delta.get(delta_field), str):
                        text_block_index = int(payload.get("index", text_block_index) or 0)
                        text_delta_type = delta_type
                        async for item in self._handle_messages_text_delta(
                            delta[delta_field],
                            text_block_index,
                            chunk,
                            delta_type,
                            state={
                                "text_buffer": text_buffer,
                                "unflushed_text": unflushed_text,
                                "pending": pending,
                                "dsml_mode": dsml_mode,
                                "raw_think": raw_think,
                            },
                        ):
                            if isinstance(item, dict) and item.get("_state"):
                                text_buffer = item["text_buffer"]
                                unflushed_text = item["unflushed_text"]
                                pending = item["pending"]
                                dsml_mode = item["dsml_mode"]
                                raw_think = item["raw_think"]
                                passthrough_blocked = item["passthrough_blocked"]
                            else:
                                yield item
                        continue
                    if (
                        stop_after_first_native_tool
                        and native_tool_index is not None
                        and int(payload.get("index", -1) or -1) == native_tool_index
                        and delta.get("type") == "input_json_delta"
                        and isinstance(delta.get("partial_json"), str)
                    ):
                        native_tool_json += delta["partial_json"]

                if dsml_mode or passthrough_blocked:
                    continue

                if event_name == "content_block_start":
                    text_block_index = int(payload.get("index", text_block_index) or 0)
                    content_block = payload.get("content_block") or {}
                    if content_block.get("type") == "tool_use" and native_tool_index is None:
                        native_tool_index = text_block_index
                elif event_name in {"content_block_stop", "message_delta", "message_stop"}:
                    tail = _flush_raw_think_tail(raw_think)
                    if tail:
                        unflushed_text += tail
                    for item in pending:
                        yield _messages_text_delta(item, text_block_index, chunk, text_delta_type)
                    pending = []
                    if unflushed_text:
                        yield _messages_text_delta(unflushed_text, text_block_index, chunk, text_delta_type)
                        unflushed_text = ""
                    text_buffer = ""

                yield _encode_like(raw_event + "\n\n", chunk)

                if (
                    stop_after_first_native_tool
                    and native_tool_index is not None
                    and event_name == "content_block_delta"
                    and _is_complete_json_object(native_tool_json)
                ):
                    yield _sse("content_block_stop", {"type": "content_block_stop", "index": native_tool_index}, chunk)
                    yield _sse(
                        "message_delta",
                        {
                            "type": "message_delta",
                            "delta": {"stop_reason": "tool_use", "stop_sequence": None},
                            "usage": {"output_tokens": 0},
                        },
                        chunk,
                    )
                    yield _sse("message_stop", {"type": "message_stop"}, chunk)
                    return

                if (
                    stop_after_first_native_tool
                    and native_tool_index is not None
                    and event_name == "content_block_stop"
                    and int(payload.get("index", -1) or -1) == native_tool_index
                ):
                    yield _sse(
                        "message_delta",
                        {
                            "type": "message_delta",
                            "delta": {"stop_reason": "tool_use", "stop_sequence": None},
                            "usage": {"output_tokens": 0},
                        },
                        chunk,
                    )
                    yield _sse("message_stop", {"type": "message_stop"}, chunk)
                    return

        if dsml_mode:
            idx = find_raw_tool_start(text_buffer)
            if idx > 0:
                yield _messages_text_delta(text_buffer[:idx], text_block_index, original_for_output, text_delta_type)
            if idx < len(text_buffer):
                yield _messages_text_delta(text_buffer[idx:], text_block_index, original_for_output, text_delta_type)
        else:
            tail = _flush_raw_think_tail(raw_think)
            if tail:
                unflushed_text += tail
            for item in pending:
                yield _messages_text_delta(item, text_block_index, original_for_output, text_delta_type)
            if unflushed_text:
                yield _messages_text_delta(unflushed_text, text_block_index, original_for_output, text_delta_type)

        if sse_buffer and not passthrough_blocked:
            yield _encode_like(sse_buffer, original_for_output)

    async def _handle_messages_text_delta(
        self,
        text: str,
        text_block_index: int,
        original: Any,
        delta_type: str,
        state: Dict[str, Any],
    ) -> AsyncGenerator[Any, None]:
        text_buffer = state["text_buffer"] + text
        unflushed_text = state["unflushed_text"]
        pending: List[str] = state["pending"]
        dsml_mode = state["dsml_mode"]
        raw_think = state["raw_think"]
        passthrough_blocked = False

        if has_complete_raw_tool_block(text_buffer):
            idx = find_raw_tool_start(text_buffer)
            if idx > 0 and not dsml_mode:
                yield _messages_text_delta(text_buffer[:idx], text_block_index, original, delta_type)

            parsed = parse_raw_tool_calls(normalize_raw_tool_calls(text_buffer))
            if parsed:
                yield _sse("content_block_stop", {"type": "content_block_stop", "index": text_block_index}, original)
                for event in _messages_tool_use_events(parsed, text_block_index + 1, original):
                    yield event
                yield {
                    "_state": True,
                    "text_buffer": "",
                    "unflushed_text": "",
                    "pending": [],
                    "dsml_mode": True,
                    "raw_think": raw_think,
                    "passthrough_blocked": True,
                }
                return

            yield _messages_text_delta(text_buffer[idx:], text_block_index, original, delta_type)
            yield {
                "_state": True,
                "text_buffer": "",
                "unflushed_text": "",
                "pending": [],
                "dsml_mode": False,
                "raw_think": raw_think,
                "passthrough_blocked": False,
            }
            return

        safe_text = _strip_raw_think_delta(text, raw_think)

        if dsml_mode:
            yield {
                "_state": True,
                "text_buffer": text_buffer,
                "unflushed_text": unflushed_text,
                "pending": pending,
                "dsml_mode": dsml_mode,
                "raw_think": raw_think,
                "passthrough_blocked": True,
            }
            return

        if has_any_dsml_prefix(text_buffer):
            dsml_mode = True
            passthrough_blocked = True
            for item in pending:
                yield _messages_text_delta(item, text_block_index, original, delta_type)
            pending = []
            if unflushed_text:
                yield _messages_text_delta(unflushed_text, text_block_index, original, delta_type)
                unflushed_text = ""
            idx = find_raw_tool_start(text_buffer)
            if idx < len(text_buffer):
                text_buffer = text_buffer[idx:]
            yield {
                "_state": True,
                "text_buffer": text_buffer,
                "unflushed_text": unflushed_text,
                "pending": pending,
                "dsml_mode": dsml_mode,
                "raw_think": raw_think,
                "passthrough_blocked": passthrough_blocked,
            }
            return

        unflushed_text += safe_text
        while len(unflushed_text) >= SECTION_SIZE:
            pending.append(unflushed_text[:SECTION_SIZE])
            unflushed_text = unflushed_text[SECTION_SIZE:]
            if len(pending) > GUARD_SECTIONS:
                yield _messages_text_delta(pending.pop(0), text_block_index, original, delta_type)

        yield {
            "_state": True,
            "text_buffer": text_buffer,
            "unflushed_text": unflushed_text,
            "pending": pending,
            "dsml_mode": dsml_mode,
            "raw_think": raw_think,
            "passthrough_blocked": passthrough_blocked,
        }


proxy_handler_instance = OpencodeCompatHandler()
