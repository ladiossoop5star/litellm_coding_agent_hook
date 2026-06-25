import asyncio
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
RAW_THINK_PREVIEW_LIMIT = 200
MESSAGES_STREAM_KEEPALIVE_SECONDS = 15.0
MESSAGES_STREAM_IDLE_TIMEOUT_SECONDS = 600.0
REVEAL_HIDDEN_THINKING_AFTER_SECONDS = 30.0
_RESPONSES_EMPTY_TOOLS_PATCHED = False


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
    if call_type in {"pass_through_endpoint", "responses", "aresponses"}:
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
    if _is_messages_stream(request_data):
        return True
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


def _chat_function_tool_from_responses_tool(tool: Dict[str, Any]) -> Dict[str, Any]:
    parameters = tool.get("parameters") or {}
    if not isinstance(parameters, dict):
        parameters = {"type": "object"}
    if "type" not in parameters:
        parameters = {**parameters, "type": "object"}
    return {
        "type": "function",
        "function": {
            "name": str(tool.get("name") or ""),
            "description": str(tool.get("description") or ""),
            "parameters": parameters,
            "strict": bool(tool.get("strict", False)),
        },
    }


def _sanitize_response_tools_for_litellm(tools: Any) -> Any:
    if not isinstance(tools, list):
        return tools
    return [tool for tool in tools if isinstance(tool, dict) and tool.get("type") == "function"]


def _sanitize_chat_tools_for_upstream(tools: Any) -> Any:
    if not isinstance(tools, list):
        return tools

    sanitized = []
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            continue
        if isinstance(tool.get("function"), dict):
            sanitized.append(tool)
            continue
        if tool.get("name"):
            sanitized.append(_chat_function_tool_from_responses_tool(tool))
    return sanitized


def _sanitize_request_tools(data: dict, call_type: str) -> None:
    if not isinstance(data, dict):
        return

    if call_type in ("responses", "aresponses") and _is_codex_compaction_request(data):
        _disable_tools_for_compaction(data)
        return

    if isinstance(data.get("tools"), list):
        if call_type in ("responses", "aresponses"):
            data["tools"] = _sanitize_response_tools_for_litellm(data["tools"])
        elif call_type in ("completion", "acompletion", "chat_completion"):
            data["tools"] = _sanitize_chat_tools_for_upstream(data["tools"])
        if isinstance(data.get("tools"), list) and not data["tools"]:
            _drop_empty_tools(data)

    optional_params = data.get("optional_params")
    if (
        call_type in ("responses", "aresponses", "completion", "acompletion", "chat_completion")
        and isinstance(optional_params, dict)
        and isinstance(optional_params.get("tools"), list)
    ):
        optional_params["tools"] = _sanitize_chat_tools_for_upstream(optional_params["tools"])
        if not optional_params["tools"]:
            _drop_empty_tools(optional_params)


def _drop_empty_tools(payload: Any) -> None:
    if not isinstance(payload, dict):
        return
    if isinstance(payload.get("tools"), list) and not payload["tools"]:
        payload.pop("tools", None)
        payload.pop("tool_choice", None)


def _disable_tools_for_compaction(payload: Any) -> None:
    if not isinstance(payload, dict):
        return
    payload.pop("tools", None)
    payload.pop("tool_choice", None)
    payload["parallel_tool_calls"] = False

    optional_params = payload.get("optional_params")
    if isinstance(optional_params, dict):
        optional_params.pop("tools", None)
        optional_params.pop("tool_choice", None)
        optional_params["parallel_tool_calls"] = False


def _metadata_dicts(payload: Any) -> Iterable[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return []

    dicts: List[Dict[str, Any]] = [payload]
    for key in ("metadata", "client_metadata", "litellm_metadata"):
        value = payload.get(key)
        if isinstance(value, dict):
            dicts.append(value)

    extra_body = payload.get("extra_body")
    if isinstance(extra_body, dict):
        dicts.append(extra_body)
        client_metadata = extra_body.get("client_metadata")
        if isinstance(client_metadata, dict):
            dicts.append(client_metadata)

    return dicts


def _codex_turn_metadata(payload: Any) -> Dict[str, Any]:
    for metadata in _metadata_dicts(payload):
        raw = metadata.get("x-codex-turn-metadata")
        if not isinstance(raw, str):
            continue
        try:
            parsed = json.loads(raw)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


def _iter_response_input_text(input_value: Any) -> Iterable[str]:
    if isinstance(input_value, str):
        yield input_value
        return

    if not isinstance(input_value, list):
        return

    for item in input_value:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if isinstance(content, str):
            yield content
            continue
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            text = part.get("text") or part.get("input_text")
            if isinstance(text, str):
                yield text


def _is_codex_compaction_request(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False

    turn_metadata = _codex_turn_metadata(payload)
    if turn_metadata.get("request_kind") == "compaction":
        return True
    if isinstance(turn_metadata.get("compaction"), dict):
        return True

    marker = "CONTEXT CHECKPOINT COMPACTION"
    return any(marker in text for text in _iter_response_input_text(payload.get("input")))


def _patch_litellm_responses_empty_tools_bridge() -> None:
    global _RESPONSES_EMPTY_TOOLS_PATCHED

    if _RESPONSES_EMPTY_TOOLS_PATCHED:
        return

    try:
        from litellm.responses.litellm_completion_transformation.transformation import (
            LiteLLMCompletionResponsesConfig,
        )

        original = LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request
        if getattr(original, "_opencode_empty_tools_patched", False):
            _RESPONSES_EMPTY_TOOLS_PATCHED = True
            return

        def patched_transform(*args: Any, **kwargs: Any) -> dict:
            completion_request = original(*args, **kwargs)
            _drop_empty_tools(completion_request)
            return completion_request

        setattr(patched_transform, "_opencode_empty_tools_patched", True)
        LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request = staticmethod(
            patched_transform
        )
        _RESPONSES_EMPTY_TOOLS_PATCHED = True
        log.info("patched LiteLLM Responses bridge to omit empty tools")
    except Exception as exc:
        log.warning("failed to patch LiteLLM Responses empty-tools bridge: %s", exc)


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


def _sse_comment(text: str, original: Any) -> Any:
    return _encode_like(": " + text + "\n\n", original)


async def _iter_with_keepalive(response: Any, request_context: str = "unknown-request") -> AsyncGenerator[Any, None]:
    iterator = response.__aiter__()
    next_chunk = asyncio.create_task(iterator.__anext__())
    original_for_output: Any = b""
    last_chunk_at = time.time()

    try:
        while True:
            done, _ = await asyncio.wait({next_chunk}, timeout=MESSAGES_STREAM_KEEPALIVE_SECONDS)
            if not done:
                idle_seconds = time.time() - last_chunk_at
                if idle_seconds >= MESSAGES_STREAM_IDLE_TIMEOUT_SECONDS:
                    log.warning(
                        "messages stream idle timeout after %.1fs context=%s",
                        idle_seconds,
                        request_context,
                    )
                    break
                yield _sse_comment("opencode-compat keepalive", original_for_output)
                continue

            try:
                chunk = next_chunk.result()
            except StopAsyncIteration:
                break

            original_for_output = chunk
            last_chunk_at = time.time()
            next_chunk = asyncio.create_task(iterator.__anext__())
            yield chunk
    finally:
        if not next_chunk.done():
            next_chunk.cancel()


def _is_complete_json_object(text: str) -> bool:
    try:
        return isinstance(json.loads(text), dict)
    except Exception:
        return False


def _event_index(payload: Dict[str, Any], default: int = 0) -> int:
    value = payload.get("index", default)
    if value is None:
        return default
    try:
        return int(value)
    except Exception:
        return default


def _first_complete_openai_tool_call(
    state: Dict[int, Dict[str, str]], delta: Any
) -> Optional[Dict[str, Any]]:
    tool_calls = _get(delta, "tool_calls", None)
    if not tool_calls:
        return None

    for tool_call in tool_calls:
        index = _get(tool_call, "index", 0) or 0
        entry = state.setdefault(int(index), {"id": "", "name": "", "arguments": ""})
        tool_id = _get(tool_call, "id", None)
        if tool_id:
            entry["id"] = str(tool_id)

        function = _get(tool_call, "function", {}) or {}
        name = _get(function, "name", None)
        if name:
            entry["name"] = str(name)
        arguments = _get(function, "arguments", None)
        if isinstance(arguments, str):
            entry["arguments"] += arguments

        if entry["name"] and _is_complete_json_object(entry["arguments"]):
            return {
                "id": entry["id"],
                "type": "function",
                "function": {
                    "name": entry["name"],
                    "arguments": entry["arguments"],
                },
            }

    return None


def _messages_text_delta(text: str, index: int, original: Any, delta_type: str = "text_delta") -> Any:
    field = "thinking" if delta_type == "thinking_delta" else "text"
    return _sse(
        "content_block_delta",
        {"type": "content_block_delta", "index": index, "delta": {"type": delta_type, field: text}},
        original,
    )


def _raw_think_state() -> Dict[str, Any]:
    return {
        "in_think": False,
        "tail": "",
        "started_at": None,
        "suppressed_chars": 0,
        "suppressed_chunks": 0,
        "preview": "",
        "visible_chars": 0,
        "warned_unclosed": False,
        "placeholder_emitted": False,
        "revealing": False,
        "reveal_prefix_emitted": False,
    }


def _record_raw_think_suppressed(text: str, state: Dict[str, Any]) -> None:
    if not text:
        return

    if state.get("started_at") is None:
        state["started_at"] = time.time()
    state["suppressed_chars"] = int(state.get("suppressed_chars") or 0) + len(text)
    state["suppressed_chunks"] = int(state.get("suppressed_chunks") or 0) + 1

    preview = str(state.get("preview") or "")
    if len(preview) < RAW_THINK_PREVIEW_LIMIT:
        remaining = RAW_THINK_PREVIEW_LIMIT - len(preview)
        state["preview"] = preview + text[:remaining]


def _should_reveal_hidden_thinking(state: Dict[str, Any]) -> bool:
    if state.get("revealing"):
        return True
    started_at = state.get("started_at")
    if not isinstance(started_at, (int, float)):
        return False
    if time.time() - started_at < REVEAL_HIDDEN_THINKING_AFTER_SECONDS:
        return False
    state["revealing"] = True
    log.warning(
        "revealing hidden thinking after %.1fs chars=%s chunks=%s preview=%r",
        time.time() - started_at,
        state.get("suppressed_chars") or 0,
        state.get("suppressed_chunks") or 0,
        str(state.get("preview") or "").replace("\n", "\\n"),
    )
    return True


def _hidden_thinking_reveal_prefix(state: Dict[str, Any]) -> str:
    if state.get("reveal_prefix_emitted"):
        return ""
    state["reveal_prefix_emitted"] = True
    preview = str(state.get("preview") or "")
    if not preview:
        return ""
    omitted = int(state.get("suppressed_chars") or 0) - len(preview)
    if omitted > 0:
        return preview + "\n...\n"
    return preview


def _hidden_thinking_final_fallback(
    state: Dict[str, Any], pending: Iterable[str], unflushed_text: str
) -> str:
    if _raw_think_has_visible_output(state, pending, unflushed_text):
        return ""
    if int(state.get("suppressed_chars") or 0) <= 0:
        return ""
    state["revealing"] = True
    fallback = _hidden_thinking_reveal_prefix(state)
    state["visible_chars"] = int(state.get("visible_chars") or 0) + len(fallback)
    return fallback


def _request_context(request_data: Optional[dict]) -> str:
    names = sorted(_request_model_names(request_data))
    metadata = (request_data or {}).get("litellm_metadata") or {}
    pieces = []
    if names:
        pieces.append("models=" + ",".join(names))
    for key in ("request_id", "litellm_call_id", "model_group", "deployment"):
        value = metadata.get(key)
        if value:
            pieces.append(f"{key}={value}")
    return " ".join(pieces) or "unknown-request"


def _warn_unclosed_raw_think(state: Dict[str, Any], context: str) -> None:
    if not state.get("in_think") or state.get("warned_unclosed"):
        return

    started_at = state.get("started_at")
    duration = time.time() - started_at if isinstance(started_at, (int, float)) else 0.0
    preview = str(state.get("preview") or "").replace("\n", "\\n")
    log.warning(
        "unclosed raw <think> suppressed context=%s chars=%s chunks=%s duration=%.1fs preview=%r",
        context,
        state.get("suppressed_chars") or 0,
        state.get("suppressed_chunks") or 0,
        duration,
        preview,
    )
    state["warned_unclosed"] = True


def _raw_think_placeholder(state: Dict[str, Any], context: str) -> str:
    if not state.get("in_think"):
        return ""
    _warn_unclosed_raw_think(state, context)
    if state.get("placeholder_emitted") or int(state.get("visible_chars") or 0) > 0:
        return ""
    state["placeholder_emitted"] = True
    return ASSISTANT_PLACEHOLDER


def _raw_think_has_visible_output(
    state: Dict[str, Any], pending: Iterable[str], unflushed_text: str
) -> bool:
    return (
        int(state.get("visible_chars") or 0) > 0
        or any(bool(item) for item in pending)
        or bool(unflushed_text)
    )


def _raise_empty_unclosed_raw_think(
    state: Dict[str, Any],
    context: str,
    pending: Iterable[str],
    unflushed_text: str,
    has_native_tool: bool,
) -> None:
    if not state.get("in_think") or has_native_tool:
        return

    _warn_unclosed_raw_think(state, context)
    if _raw_think_has_visible_output(state, pending, unflushed_text):
        return

    raise RuntimeError(
        "malformed model output: unclosed raw <think> produced an empty assistant turn "
        f"({context})"
    )


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
                tail = _matching_prefix_suffix(data, close_marker)
                hidden_segment = data[:-len(tail)] if tail else data
                if _should_reveal_hidden_thinking(state):
                    state["_revealed_delta"] = True
                    output.append(_hidden_thinking_reveal_prefix(state))
                    output.append(hidden_segment)
                else:
                    _record_raw_think_suppressed(hidden_segment, state)
                if tail:
                    state["tail"] = tail
                visible = "".join(output)
                state["visible_chars"] = int(state.get("visible_chars") or 0) + len(visible)
                return visible
            hidden_segment = data[:close_idx]
            if _should_reveal_hidden_thinking(state):
                state["_revealed_delta"] = True
                output.append(_hidden_thinking_reveal_prefix(state))
                output.append(hidden_segment)
            else:
                _record_raw_think_suppressed(hidden_segment, state)
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
            visible = "".join(output)
            state["visible_chars"] = int(state.get("visible_chars") or 0) + len(visible)
            return visible

        output.append(data[:open_idx])
        data = data[open_idx + len(open_marker):]
        state["in_think"] = True
        state["started_at"] = time.time()

    visible = "".join(output)
    state["visible_chars"] = int(state.get("visible_chars") or 0) + len(visible)
    return visible


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


def _messages_end_turn_events(index: int, original: Any) -> List[Any]:
    return [
        _sse("content_block_stop", {"type": "content_block_stop", "index": index}, original),
        _sse(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 0},
            },
            original,
        ),
        _sse("message_stop", {"type": "message_stop"}, original),
    ]


class OpencodeCompatHandler(CustomLogger):
    """Compatibility layer for opencode raw DSML/Qwen tool-call output."""

    def __init__(self) -> None:
        _patch_litellm_responses_empty_tools_bridge()
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
        _sanitize_request_tools(data, call_type)

        if call_type not in ("completion", "acompletion", "chat_completion", "anthropic_messages"):
            return data

        _normalize_assistant_messages(data.get("messages"))
        return data

    async def async_pre_request_hook(self, model: str, messages: List[Any], kwargs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if _is_codex_compaction_request(kwargs):
            _disable_tools_for_compaction(kwargs)
            return kwargs

        if isinstance(kwargs, dict) and isinstance(kwargs.get("tools"), list):
            kwargs["tools"] = _sanitize_chat_tools_for_upstream(kwargs["tools"])
            _drop_empty_tools(kwargs)
            return kwargs
        return None

    async def async_post_call_success_hook(self, data: dict, user_api_key_dict: Any, response: Any) -> Any:
        return convert_non_streaming_response(response)

    async def async_post_call_streaming_iterator_hook(
        self, user_api_key_dict: Any, response: Any, request_data: dict
    ) -> AsyncGenerator[Any, None]:
        if _is_messages_stream(request_data):
            async for chunk in self._convert_anthropic_messages_stream(
                response,
                stop_after_first_native_tool=_stop_after_first_native_tool(request_data),
                request_context=_request_context(request_data),
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
        raw_stream_passthrough = False
        thinking_state = 0
        last_id = "chatcmpl-opencode-compat"
        last_model = request_data.get("model", "unknown") if request_data else "unknown"
        last_created = int(time.time())

        async for chunk in response:
            # Native passthrough streams can be bytes; leave them untouched.
            if isinstance(chunk, (bytes, bytearray)):
                raw_stream_passthrough = True
                yield chunk
                continue

            if isinstance(chunk, str) and (chunk.startswith("data:") or chunk.startswith("event:")):
                raw_stream_passthrough = True
                yield chunk
                continue

            event_type = _get(chunk, "type", None)
            if isinstance(event_type, str) and event_type.startswith("response."):
                raw_stream_passthrough = True
                yield chunk
                continue

            chunk_as_text = str(chunk)
            if chunk_as_text.startswith("data:") or chunk_as_text.startswith("event:"):
                raw_stream_passthrough = True
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

            previous_buffer_len = len(buffer)
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
                    log.warning("suppressing unparsable stream raw tool block: %s", buffer[idx:idx + 800])
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
                idx = find_raw_tool_start(buffer)
                if idx > previous_buffer_len:
                    yield _make_content_chunk(last_id, last_model, last_created, buffer[previous_buffer_len:idx])
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
                    log.warning("suppressing incomplete stream raw tool block: %s", buffer[idx:idx + 800])
        else:
            for item in pending:
                yield _make_content_chunk(last_id, last_model, last_created, item)
            if unflushed_text:
                yield _make_content_chunk(last_id, last_model, last_created, unflushed_text)

        if raw_stream_passthrough and not content_collected:
            return

        yield _make_stream_chunk(last_id, last_model, last_created, {"content": ""}, finish_reason="stop")

    async def _convert_anthropic_messages_stream(
        self,
        response: Any,
        stop_after_first_native_tool: bool = False,
        request_context: str = "unknown-request",
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
        open_content_blocks: set[int] = set()
        saw_content_block = False
        openai_tool_state: Dict[int, Dict[str, str]] = {}
        saw_message_stop = False
        saw_stop_message_delta = False
        synthetic_stop_sent = False

        async for chunk in _iter_with_keepalive(response, request_context):
            original_for_output = chunk
            if stop_after_first_native_tool:
                complete_openai_tool = _first_complete_openai_tool_call(openai_tool_state, _delta(chunk))
                if complete_openai_tool:
                    for index in sorted(open_content_blocks):
                        yield _sse("content_block_stop", {"type": "content_block_stop", "index": index}, chunk)
                    tool_index = max(open_content_blocks | {text_block_index}) + 1 if saw_content_block else 0
                    for event in _messages_tool_use_events([complete_openai_tool], tool_index, chunk):
                        yield event
                    log.info("synthesized messages native tool stop from OpenAI stream context=%s", request_context)
                    return

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
                        text_block_index = _event_index(payload, text_block_index)
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
                                synthetic_stop_sent = synthetic_stop_sent or bool(item.get("stop_sent"))
                            else:
                                yield item
                        continue
                    if (
                        stop_after_first_native_tool
                        and native_tool_index is not None
                        and _event_index(payload, -1) == native_tool_index
                        and delta.get("type") == "input_json_delta"
                        and isinstance(delta.get("partial_json"), str)
                    ):
                        native_tool_json += delta["partial_json"]

                if dsml_mode or passthrough_blocked:
                    continue

                if event_name == "content_block_start":
                    saw_content_block = True
                    text_block_index = _event_index(payload, text_block_index)
                    open_content_blocks.add(text_block_index)
                    content_block = payload.get("content_block") or {}
                    if content_block.get("type") == "tool_use" and native_tool_index is None:
                        native_tool_index = text_block_index
                elif event_name == "content_block_stop":
                    open_content_blocks.discard(_event_index(payload, text_block_index))

                if event_name in {"content_block_stop", "message_delta", "message_stop"}:
                    tail = _flush_raw_think_tail(raw_think)
                    if tail:
                        unflushed_text += tail
                    fallback = _hidden_thinking_final_fallback(raw_think, pending, unflushed_text)
                    if fallback:
                        yield _messages_text_delta(fallback, text_block_index, chunk, "text_delta")
                    if event_name == "message_delta":
                        delta = payload.get("delta") or {}
                        stop_reason = delta.get("stop_reason")
                        if stop_reason:
                            saw_stop_message_delta = True
                        if stop_reason == "end_turn":
                            _raise_empty_unclosed_raw_think(
                                raw_think,
                                request_context,
                                pending,
                                unflushed_text,
                                native_tool_index is not None,
                            )
                        elif raw_think.get("in_think"):
                            _warn_unclosed_raw_think(raw_think, request_context)
                    elif event_name == "message_stop" and raw_think.get("in_think"):
                        _warn_unclosed_raw_think(raw_think, request_context)
                    if event_name == "message_stop":
                        saw_message_stop = True
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
                    log.info("synthesized messages native tool stop from Anthropic SSE context=%s", request_context)
                    return

                if (
                    stop_after_first_native_tool
                    and native_tool_index is not None
                    and event_name == "content_block_stop"
                    and _event_index(payload, -1) == native_tool_index
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
                log.warning(
                    "suppressing incomplete messages raw tool block context=%s preview=%r",
                    request_context,
                    text_buffer[idx:idx + 800].replace("\n", "\\n"),
                )
        else:
            tail = _flush_raw_think_tail(raw_think)
            if tail:
                unflushed_text += tail
            if raw_think.get("in_think"):
                _warn_unclosed_raw_think(raw_think, request_context)
            for item in pending:
                yield _messages_text_delta(item, text_block_index, original_for_output, text_delta_type)
            if unflushed_text:
                yield _messages_text_delta(unflushed_text, text_block_index, original_for_output, text_delta_type)

        if sse_buffer and not passthrough_blocked:
            yield _encode_like(sse_buffer, original_for_output)

        if not saw_message_stop and not synthetic_stop_sent:
            for index in sorted(open_content_blocks):
                yield _sse("content_block_stop", {"type": "content_block_stop", "index": index}, original_for_output)
            if not saw_stop_message_delta:
                yield _sse(
                    "message_delta",
                    {
                        "type": "message_delta",
                        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                        "usage": {"output_tokens": 0},
                    },
                    original_for_output,
                )
            yield _sse("message_stop", {"type": "message_stop"}, original_for_output)
            log.warning("synthesized missing messages stream stop context=%s", request_context)

    async def _handle_messages_text_delta(
        self,
        text: str,
        text_block_index: int,
        original: Any,
        delta_type: str,
        state: Dict[str, Any],
    ) -> AsyncGenerator[Any, None]:
        text_buffer = state["text_buffer"]
        unflushed_text = state["unflushed_text"]
        pending: List[str] = state["pending"]
        dsml_mode = state["dsml_mode"]
        raw_think = state["raw_think"]
        passthrough_blocked = False

        probe_buffer = text_buffer + text
        should_parse_tool = (
            dsml_mode
            or has_complete_raw_tool_block(probe_buffer)
            or has_any_dsml_prefix(probe_buffer)
        )

        if delta_type == "thinking_delta":
            if not should_parse_tool:
                if raw_think.get("started_at") is None:
                    raw_think["started_at"] = time.time()
                if _should_reveal_hidden_thinking(raw_think):
                    visible_text = _hidden_thinking_reveal_prefix(raw_think) + text
                    raw_think["visible_chars"] = int(raw_think.get("visible_chars") or 0) + len(visible_text)
                    yield _messages_text_delta(visible_text, text_block_index, original, "text_delta")
                else:
                    _record_raw_think_suppressed(text, raw_think)
                    yield _messages_text_delta(text, text_block_index, original, "thinking_delta")
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

        raw_think["_revealed_delta"] = False
        if should_parse_tool:
            safe_text = text
            revealed_hidden_delta = False
        else:
            safe_text = _strip_raw_think_delta(text, raw_think)
            revealed_hidden_delta = bool(raw_think.pop("_revealed_delta", False))
        previous_text_len = len(text_buffer)
        text_buffer += safe_text

        if not revealed_hidden_delta and has_complete_raw_tool_block(text_buffer):
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
                    "stop_sent": True,
                }
                return

            log.warning(
                "suppressing unparsable messages raw tool block preview=%r",
                text_buffer[idx:idx + 800].replace("\n", "\\n"),
            )
            for event in _messages_end_turn_events(text_block_index, original):
                yield event
            yield {
                "_state": True,
                "text_buffer": "",
                "unflushed_text": "",
                "pending": [],
                "dsml_mode": True,
                "raw_think": raw_think,
                "passthrough_blocked": True,
                "stop_sent": True,
            }
            return

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

        if not revealed_hidden_delta and has_any_dsml_prefix(text_buffer):
            dsml_mode = True
            passthrough_blocked = True
            for item in pending:
                yield _messages_text_delta(item, text_block_index, original, delta_type)
            pending = []
            if unflushed_text:
                yield _messages_text_delta(unflushed_text, text_block_index, original, delta_type)
                unflushed_text = ""
            idx = find_raw_tool_start(text_buffer)
            if idx > previous_text_len:
                yield _messages_text_delta(text_buffer[previous_text_len:idx], text_block_index, original, delta_type)
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
