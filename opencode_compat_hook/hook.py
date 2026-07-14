import asyncio
import fcntl
import json
import logging
import re
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
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s - opencode_compat_hook - %(levelname)s - %(message)s"))
    log.addHandler(_h)
    log.setLevel(logging.INFO)

SECTION_SIZE = 32
GUARD_SECTIONS = 2
ASSISTANT_PLACEHOLDER = "."
STOP_AFTER_FIRST_NATIVE_TOOL_MODEL_MARKER = "deepseek"
RAW_THINK_PREVIEW_LIMIT = 200
MESSAGES_STREAM_KEEPALIVE_SECONDS = 15.0
MESSAGES_STREAM_IDLE_TIMEOUT_SECONDS = 600.0
REVEAL_HIDDEN_THINKING_AFTER_SECONDS = 30.0
STOP_HOOK_KEEPALIVE_SECONDS = 5.0
STOP_HOOK_JSON_FALLBACK_SECONDS = 28.0
STOP_HOOK_JSON_FALLBACK_MAX_CONSECUTIVE = 5
STOP_HOOK_JSON_FALLBACK_IDLE_RESET_SECONDS = 30 * 60
COUNT_TOKENS_NATIVE_MAX_ESTIMATE = 8192
_RESPONSES_EMPTY_TOOLS_PATCHED = False
_RESPONSES_REASONING_TEXT_PATCHED = False
_STOP_HOOK_JSON_FALLBACK_COUNTS_PATH = "/tmp/opencode_compat_stop_hook_fallback_counts.json"
_STOP_HOOK_JSON_FALLBACK_COUNTS: Dict[str, int] = {}
_STOP_HOOK_REQUEST_STARTED_AT: Dict[str, float] = {}


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


def _patch_tool_call_ids(delta: Any, state: Dict[Any, Dict[str, str]]) -> None:
    tool_calls = _get(delta, "tool_calls", None)
    if not tool_calls:
        return

    for position, tc in enumerate(tool_calls):
        index = _get(tc, "index", None)
        state_key = index if index is not None else position
        tid = _get(tc, "id", None)
        fn = _get(tc, "function", None)
        name = _get(fn, "name", None) if fn is not None else None
        entry = state.get(state_key)

        if tid and (entry is None or entry.get("id") != tid):
            entry = {"id": str(tid)}
            state[state_key] = entry

        if not tid:
            if entry and entry.get("id"):
                _set(tc, "id", entry["id"])
            else:
                tid = "call_" + uuid.uuid4().hex
                _set(tc, "id", tid)
                entry = {"id": tid}
                state[state_key] = entry

        if isinstance(name, str) and name:
            if entry is None:
                entry = {"id": str(_get(tc, "id", ""))}
                state[state_key] = entry
            entry["name"] = name
        elif fn is not None and entry and entry.get("name"):
            _set(fn, "name", entry["name"])


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


def _strip_raw_think_from_history_text(text: str) -> str:
    state = _raw_think_state()
    visible = _strip_raw_think_delta(text, state)
    visible += _flush_raw_think_tail(state)
    if state.get("in_think"):
        _warn_unclosed_raw_think(state, "responses-input-history")
    return visible


_INTERNAL_ARTIFACT_BLOCK_PATTERNS = (
    re.compile(
        r"<dcp-system-reminder\b[^>]*>.*?(?:</dcp-system-reminder>|</\uff5cDSML\uff5csystem-reminder>|</\|DSML\|system-reminder>)",
        re.DOTALL,
    ),
    re.compile(
        r"<system-reminder\b[^>]*>.*?</system-reminder>",
        re.DOTALL,
    ),
    re.compile(
        r"<(?:\uff5cDSML\uff5c|\|DSML\|)system-reminder\b[^>]*>.*?</(?:\uff5cDSML\uff5c|\|DSML\|)system-reminder>",
        re.DOTALL,
    ),
    re.compile(
        r"<dcp-message-id\b[^>]*>.*?</dcp-message-id>",
        re.DOTALL,
    ),
)


def _strip_internal_artifacts_from_history_text(text: str) -> str:
    cleaned = text
    for pattern in _INTERNAL_ARTIFACT_BLOCK_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    return cleaned


def _sanitize_content_text_parts(content: Any) -> Tuple[Any, bool]:
    if isinstance(content, str):
        cleaned = _strip_internal_artifacts_from_history_text(content)
        return cleaned, cleaned != content

    if not isinstance(content, list):
        return content, False

    changed = False
    new_content: List[Any] = []
    for part in content:
        if not isinstance(part, dict):
            new_content.append(part)
            continue

        text_key = None
        for candidate in ("text", "input_text"):
            if isinstance(part.get(candidate), str):
                text_key = candidate
                break
        if text_key is None:
            new_content.append(part)
            continue

        cleaned = _strip_internal_artifacts_from_history_text(part[text_key])
        if cleaned != part[text_key]:
            changed = True
            if not cleaned.strip():
                continue
            new_part = dict(part)
            new_part[text_key] = cleaned
            new_content.append(new_part)
        else:
            new_content.append(part)

    return new_content, changed


def _sanitize_chat_internal_artifact_history(payload: Any) -> None:
    if not isinstance(payload, dict):
        return
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return

    changed_parts = 0
    removed_messages = 0
    sanitized_messages: List[Any] = []
    for message in messages:
        if not isinstance(message, dict):
            sanitized_messages.append(message)
            continue

        content = message.get("content")
        cleaned_content, changed = _sanitize_content_text_parts(content)
        if not changed:
            sanitized_messages.append(message)
            continue

        changed_parts += 1
        if isinstance(cleaned_content, str) and not cleaned_content.strip():
            removed_messages += 1
            continue
        if isinstance(cleaned_content, list) and not cleaned_content:
            removed_messages += 1
            continue

        new_message = dict(message)
        new_message["content"] = cleaned_content
        sanitized_messages.append(new_message)

    if changed_parts or removed_messages:
        payload["messages"] = sanitized_messages
        log.warning(
            "sanitized chat internal artifact history removed_messages=%s changed_parts=%s",
            removed_messages,
            changed_parts,
        )


def _sanitize_response_input_history(payload: Any) -> None:
    if not isinstance(payload, dict):
        return
    input_value = payload.get("input")
    if not isinstance(input_value, list):
        return

    removed_items = 0
    stripped_parts = 0
    sanitized_items: List[Any] = []

    for item in input_value:
        if not isinstance(item, dict):
            sanitized_items.append(item)
            continue
        if item.get("role") != "assistant" or item.get("type") not in (None, "message"):
            sanitized_items.append(item)
            continue

        content = item.get("content")
        if isinstance(content, str):
            cleaned = _strip_raw_think_from_history_text(content)
            if cleaned.strip():
                new_item = dict(item)
                new_item["content"] = cleaned
                sanitized_items.append(new_item)
            else:
                removed_items += 1
            if cleaned != content:
                stripped_parts += 1
            continue

        if not isinstance(content, list):
            sanitized_items.append(item)
            continue

        new_content: List[Any] = []
        for part in content:
            if not isinstance(part, dict):
                new_content.append(part)
                continue
            part_type = part.get("type")
            if part_type not in {"output_text", "text", "input_text"}:
                new_content.append(part)
                continue
            text = part.get("text")
            if not isinstance(text, str):
                new_content.append(part)
                continue
            cleaned = _strip_raw_think_from_history_text(text)
            if cleaned != text:
                stripped_parts += 1
            if cleaned.strip():
                new_part = dict(part)
                new_part["text"] = cleaned
                new_content.append(new_part)

        if new_content:
            new_item = dict(item)
            new_item["content"] = new_content
            sanitized_items.append(new_item)
        else:
            removed_items += 1

    if removed_items or stripped_parts:
        payload["input"] = sanitized_items
        log.warning(
            "sanitized responses input raw <think> history removed_items=%s stripped_parts=%s",
            removed_items,
            stripped_parts,
        )


def _disable_responses_reasoning_merge(payload: Any) -> None:
    if not isinstance(payload, dict):
        return
    payload["merge_reasoning_content_in_choices"] = False
    optional_params = payload.get("optional_params")
    if isinstance(optional_params, dict):
        optional_params["merge_reasoning_content_in_choices"] = False


def _model_group_names_from_payload(payload: Any) -> set[str]:
    if not isinstance(payload, dict):
        return set()
    metadata = payload.get("litellm_metadata") or {}
    values = {
        payload.get("model"),
        metadata.get("model_group"),
        metadata.get("deployment"),
        metadata.get("deployment_model_name"),
    }
    return {str(value).lower() for value in values if value}


def _valid_tool_arguments_json(arguments: Any) -> bool:
    if not isinstance(arguments, str):
        return False
    try:
        json.loads(arguments)
        return True
    except Exception:
        return False


def _response_function_call_ids(item: Dict[str, Any]) -> set[str]:
    return {str(value) for value in (item.get("call_id"), item.get("id")) if value}


def _sanitize_malformed_function_call_history(payload: Any) -> None:
    if not isinstance(payload, dict):
        return
    input_value = payload.get("input")
    if not isinstance(input_value, list):
        return

    bad_call_ids: set[str] = set()
    malformed_calls = 0
    for item in input_value:
        if not isinstance(item, dict) or item.get("type") != "function_call":
            continue
        if _valid_tool_arguments_json(item.get("arguments")):
            continue
        malformed_calls += 1
        bad_call_ids.update(_response_function_call_ids(item))

    if not malformed_calls:
        return

    removed_outputs = 0
    sanitized_items: List[Any] = []
    for item in input_value:
        if not isinstance(item, dict):
            sanitized_items.append(item)
            continue
        item_type = item.get("type")
        if item_type == "function_call" and _response_function_call_ids(item) & bad_call_ids:
            continue
        if item_type == "function_call_output" and str(item.get("call_id") or "") in bad_call_ids:
            removed_outputs += 1
            continue
        sanitized_items.append(item)

    payload["input"] = sanitized_items
    log.warning(
        "sanitized malformed responses function_call history model=%s calls=%s outputs=%s",
        ",".join(sorted(_model_group_names_from_payload(payload))) or "unknown",
        malformed_calls,
        removed_outputs,
    )


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


def _patch_litellm_responses_reasoning_text_bridge() -> None:
    global _RESPONSES_REASONING_TEXT_PATCHED

    if _RESPONSES_REASONING_TEXT_PATCHED:
        return

    try:
        from litellm.completion_extras.litellm_responses_transformation.transformation import (
            OpenAiResponsesToChatCompletionStreamIterator,
        )

        original = OpenAiResponsesToChatCompletionStreamIterator.translate_responses_chunk_to_openai_stream
        if getattr(original, "_opencode_reasoning_text_patched", False):
            _RESPONSES_REASONING_TEXT_PATCHED = True
            return

        def patched_translate(parsed_chunk: Any) -> ModelResponseStream:
            chunk = parsed_chunk.model_dump() if hasattr(parsed_chunk, "model_dump") else parsed_chunk
            if isinstance(chunk, dict) and chunk.get("type") == "response.reasoning_text.delta":
                content_part = chunk.get("delta")
                if isinstance(content_part, str) and content_part:
                    return ModelResponseStream(
                        choices=[
                            {
                                "index": int(chunk.get("summary_index") or 0),
                                "delta": {"reasoning_content": content_part},
                                "finish_reason": None,
                            }
                        ]
                    )
            return original(parsed_chunk)

        setattr(patched_translate, "_opencode_reasoning_text_patched", True)
        OpenAiResponsesToChatCompletionStreamIterator.translate_responses_chunk_to_openai_stream = staticmethod(
            patched_translate
        )
        _RESPONSES_REASONING_TEXT_PATCHED = True
        log.info("patched LiteLLM Responses bridge to preserve reasoning_text deltas")
    except Exception as exc:
        log.warning("failed to patch LiteLLM Responses reasoning_text bridge: %s", exc)


_CLIENT_DISCONNECT_METADATA_PATCHED = False


def _patch_litellm_client_disconnect_metadata() -> None:
    global _CLIENT_DISCONNECT_METADATA_PATCHED

    if _CLIENT_DISCONNECT_METADATA_PATCHED:
        return

    try:
        from litellm.proxy import common_request_processing as _crp

        original = _crp._apply_client_disconnect_metadata
        if getattr(original, "_opencode_disconnect_patched", False):
            _CLIENT_DISCONNECT_METADATA_PATCHED = True
            return

        def _safe_apply_disconnect(target_metadata: Any) -> None:
            if not isinstance(target_metadata, dict):
                return
            target_metadata["client_disconnected"] = True
            target_metadata["error_information"] = dict(_crp._CLIENT_DISCONNECTED_ERROR_INFORMATION)

        setattr(_safe_apply_disconnect, "_opencode_disconnect_patched", True)
        _crp._apply_client_disconnect_metadata = _safe_apply_disconnect
        _CLIENT_DISCONNECT_METADATA_PATCHED = True
        log.info("patched LiteLLM _apply_client_disconnect_metadata to handle None metadata")
    except Exception as exc:
        log.warning("failed to patch LiteLLM client disconnect metadata: %s", exc)


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


async def _iter_with_keepalive(
    response: Any,
    request_context: str = "unknown-request",
    keepalive_seconds: float = MESSAGES_STREAM_KEEPALIVE_SECONDS,
    force_keepalive_at: Optional[float] = None,
) -> AsyncGenerator[Any, None]:
    iterator = response.__aiter__()
    next_chunk = asyncio.create_task(iterator.__anext__())
    original_for_output: Any = b""
    last_chunk_at = time.time()
    forced_keepalive_sent = False

    try:
        while True:
            timeout = keepalive_seconds
            if force_keepalive_at is not None and not forced_keepalive_sent:
                timeout = min(timeout, max(0.0, force_keepalive_at - time.time()))
            done, _ = await asyncio.wait({next_chunk}, timeout=timeout)
            if not done:
                idle_seconds = time.time() - last_chunk_at
                if idle_seconds >= MESSAGES_STREAM_IDLE_TIMEOUT_SECONDS:
                    log.warning(
                        "messages stream idle timeout after %.1fs context=%s",
                        idle_seconds,
                        request_context,
                    )
                    break
                if force_keepalive_at is not None and time.time() >= force_keepalive_at:
                    forced_keepalive_sent = True
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
        elif not next_chunk.cancelled():
            try:
                next_chunk.exception()
            except Exception:
                pass


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

        log.info("openai_tool_delta index=%s id=%s name=%r args_preview=%s",
                 index, _get(tool_call, "id", None),
                 name, str(arguments or "")[:200])

        if entry["name"] and _is_complete_json_object(entry["arguments"]):
            log.info("openai_tool_completed name=%r args=%s", entry["name"], entry["arguments"][:200])
            return {
                "id": entry["id"] or "call_" + uuid.uuid4().hex,
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


def _raw_think_preview_has_tool_prefix(state: Dict[str, Any]) -> bool:
    preview = str(state.get("preview") or "")
    if not preview:
        return False
    lowered = preview.lower()
    return (
        "<tool_call" in lowered
        or "<｜dsml｜tool_calls" in lowered
        or "<|dsml|tool_calls" in lowered
        or has_any_dsml_prefix(preview)
    )


def _hidden_thinking_final_fallback(
    state: Dict[str, Any], pending: Iterable[str], unflushed_text: str
) -> str:
    if state.get("in_think") and _raw_think_preview_has_tool_prefix(state):
        return ""
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


def _stop_hook_session_key(request_data: Optional[dict], request_context: str) -> str:
    if not request_data:
        return request_context
    metadata = request_data.get("litellm_metadata") or {}
    headers = metadata.get("headers") or {}
    if isinstance(headers, dict):
        for key in ("x-claude-code-session-id", "session-id", "x-client-request-id"):
            value = headers.get(key)
            if value:
                return str(value)
    for key in ("session_id", "trace_id"):
        value = metadata.get(key)
        if value:
            return str(value)
    for text in _iter_nested_strings(request_data.get("metadata") or {}):
        if "session_id" not in text:
            continue
        try:
            parsed = json.loads(text)
        except Exception:
            continue
        if isinstance(parsed, dict) and parsed.get("session_id"):
            return str(parsed["session_id"])
    return request_context


def _stop_hook_request_key(session_key: str) -> str:
    return session_key


def _iter_nested_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
        return
    if isinstance(value, dict):
        for item in value.values():
            yield from _iter_nested_strings(item)
        return
    if isinstance(value, list):
        for item in value:
            yield from _iter_nested_strings(item)


def _has_stop_hook_marker(value: Any) -> bool:
    for text in _iter_nested_strings(value):
        if "hook_event_name" in text and "Stop" in text:
            return True
    return False


def _has_stop_hook_json_schema(value: Any) -> bool:
    if isinstance(value, dict):
        required = value.get("required")
        if isinstance(required, list) and {"ok", "reason", "impossible"}.issubset(set(required)):
            return True
        for item in value.values():
            if _has_stop_hook_json_schema(item):
                return True
    elif isinstance(value, list):
        return any(_has_stop_hook_json_schema(item) for item in value)
    return False


def _has_stop_hook_condition_prompt(value: Any) -> bool:
    for text in _iter_nested_strings(value):
        if (
            "stopping condition" in text
            and "hook_event_name" in text
            and "Stop" in text
            and "ARGUMENTS" in text
        ):
            return True
    return False


def _is_stop_hook_json_evaluator(request_data: Optional[dict]) -> bool:
    if not _is_messages_stream(request_data):
        return False
    if not request_data:
        return False
    return _has_stop_hook_marker(request_data) and (
        _has_stop_hook_json_schema(request_data) or _has_stop_hook_condition_prompt(request_data)
    )


def _stop_hook_json_fallback_text() -> str:
    return json.dumps(
        {
            "ok": False,
            "reason": (
                "No usable Stop hook JSON was produced by the upstream model; "
                "continue because the stopping condition is not proven satisfied."
            ),
            "impossible": False,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _is_valid_stop_hook_json_text(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    try:
        parsed = json.loads(stripped)
    except Exception:
        return False
    if not isinstance(parsed, dict):
        return False
    if not isinstance(parsed.get("ok"), bool):
        return False
    if not isinstance(parsed.get("reason"), str):
        return False
    impossible = parsed.get("impossible")
    return impossible is None or isinstance(impossible, bool)


def _mutate_stop_hook_fallback_counts(mutator: Any) -> Any:
    try:
        with open(_STOP_HOOK_JSON_FALLBACK_COUNTS_PATH, "a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.seek(0)
            raw = handle.read().strip()
            counts = json.loads(raw) if raw else {}
            if not isinstance(counts, dict):
                counts = {}
            result = mutator(counts)
            handle.seek(0)
            handle.truncate()
            json.dump(counts, handle, ensure_ascii=False, separators=(",", ":"))
            handle.flush()
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            return result
    except Exception as exc:
        log.warning("Stop hook fallback counter file unavailable, using process memory: %s", exc)
        return mutator(_STOP_HOOK_JSON_FALLBACK_COUNTS)


def _stop_hook_fallback_entry(value: Any) -> Tuple[int, Optional[float]]:
    if isinstance(value, dict):
        try:
            count = int(value.get("count") or 0)
        except Exception:
            count = 0
        updated_at = value.get("updated_at")
        if isinstance(updated_at, (int, float)):
            return count, float(updated_at)
        return count, None
    try:
        return int(value or 0), None
    except Exception:
        return 0, None


def _stop_hook_fallback_entry_expired(updated_at: Optional[float], now: float) -> bool:
    return updated_at is not None and now - updated_at > STOP_HOOK_JSON_FALLBACK_IDLE_RESET_SECONDS


def _set_stop_hook_fallback_entry(counts: Dict[str, Any], session_key: str, count: int, now: float) -> None:
    counts[session_key] = {"count": count, "updated_at": now}


def _stop_hook_json_fallback_available(session_key: str, request_context: str) -> bool:
    now = time.time()

    def mutate(counts: Dict[str, Any]) -> Tuple[bool, int, bool]:
        count, updated_at = _stop_hook_fallback_entry(counts.get(session_key))
        if _stop_hook_fallback_entry_expired(updated_at, now):
            counts.pop(session_key, None)
            return True, 0, True
        if count < STOP_HOOK_JSON_FALLBACK_MAX_CONSECUTIVE:
            return True, count, False
        counts.pop(session_key, None)
        return False, count, False

    available, count, idle_reset = _mutate_stop_hook_fallback_counts(mutate)
    if idle_reset:
        log.info(
            "Stop hook JSON fallback counter reset after %.0fs idle session=%s context=%s",
            STOP_HOOK_JSON_FALLBACK_IDLE_RESET_SECONDS,
            session_key,
            request_context,
        )
    if available:
        return True
    log.warning(
        "Stop hook JSON fallback suppressed after %d consecutive fallbacks and counter reset session=%s context=%s",
        count,
        session_key,
        request_context,
    )
    return False


def _record_stop_hook_json_fallback(session_key: str, request_context: str, reason: str) -> None:
    now = time.time()

    def mutate(counts: Dict[str, Any]) -> int:
        count, updated_at = _stop_hook_fallback_entry(counts.get(session_key))
        if _stop_hook_fallback_entry_expired(updated_at, now):
            count = 0
        count += 1
        _set_stop_hook_fallback_entry(counts, session_key, count, now)
        return count

    count = _mutate_stop_hook_fallback_counts(mutate)
    log.warning(
        "Stop hook JSON fallback count=%d/%d reason=%s session=%s context=%s",
        count,
        STOP_HOOK_JSON_FALLBACK_MAX_CONSECUTIVE,
        reason,
        session_key,
        request_context,
    )


def _record_stop_hook_valid_json(session_key: str, request_context: str) -> None:
    def mutate(counts: Dict[str, Any]) -> int:
        previous, _ = _stop_hook_fallback_entry(counts.pop(session_key, 0))
        return previous

    previous = _mutate_stop_hook_fallback_counts(mutate)
    if previous:
        log.info(
            "Stop hook JSON fallback counter reset after valid JSON previous=%d session=%s context=%s",
            previous,
            session_key,
            request_context,
        )


def _incomplete_raw_tool_fallback_text() -> str:
    return (
        "model output malformed: upstream ended inside an incomplete raw tool call; "
        "no usable response or tool call was produced. Retry this step."
    )


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
        log.info("emitting_tool_use name=%r tool_id=%s args_preview=%s", fn["name"], tool_id, str(args)[:200])
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


def _messages_text_end_turn_events(text: str, index: int, original: Any, start_block: bool = False) -> List[Any]:
    events: List[Any] = []
    if start_block:
        events.append(
            _sse(
                "content_block_start",
                {"type": "content_block_start", "index": index, "content_block": {"type": "text", "text": ""}},
                original,
            )
        )
    events.append(_messages_text_delta(text, index, original, "text_delta"))
    events.extend(_messages_end_turn_events(index, original))
    return events


def _estimate_count_tokens(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        cjk_chars = sum(
            1
            for char in value
            if "\u3400" <= char <= "\u9fff"
            or "\uf900" <= char <= "\ufaff"
            or "\u3040" <= char <= "\u30ff"
            or "\uac00" <= char <= "\ud7af"
        )
        other_chars = len(value) - cjk_chars
        return cjk_chars + (other_chars + 3) // 4
    if isinstance(value, (int, float, bool)):
        return 1
    if isinstance(value, dict):
        return 4 + sum(_estimate_count_tokens(k) + _estimate_count_tokens(v) for k, v in value.items())
    if isinstance(value, list):
        return 2 + sum(_estimate_count_tokens(item) for item in value)
    return (len(str(value)) + 2) // 3


def _estimate_anthropic_messages_tokens(payload: Dict[str, Any]) -> int:
    total = 0
    total += _estimate_count_tokens(payload.get("system"))
    total += _estimate_count_tokens(payload.get("messages"))
    total += _estimate_count_tokens(payload.get("tools"))
    total += _estimate_count_tokens(payload.get("tool_choice"))
    # Account for role/content framing and request metadata that local tokenizers
    # do not see but model chat templates do.
    message_count = len(payload.get("messages") or []) if isinstance(payload.get("messages"), list) else 0
    tool_count = len(payload.get("tools") or []) if isinstance(payload.get("tools"), list) else 0
    total += 128 + message_count * 12 + tool_count * 24
    return max(1, int(total * 1.10))


def _extract_input_tokens(value: Any) -> Optional[int]:
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    if hasattr(value, "body"):
        try:
            value = json.loads(value.body)
        except Exception:
            return None
    if isinstance(value, dict):
        tokens = value.get("input_tokens", value.get("total_tokens"))
        try:
            return int(tokens)
        except Exception:
            return None
    return None


def _message_start_with_estimated_usage(payload: Dict[str, Any], request_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    message = payload.get("message")
    if not isinstance(message, dict):
        return payload
    usage = message.get("usage")
    usage = usage if isinstance(usage, dict) else {}
    try:
        if int(usage.get("input_tokens") or 0) > 0:
            return payload
    except Exception:
        pass

    estimated_tokens = _estimate_anthropic_messages_tokens(request_data or {})
    patched_payload = dict(payload)
    patched_message = dict(message)
    patched_usage = dict(usage)
    patched_usage["input_tokens"] = estimated_tokens
    patched_usage.setdefault("output_tokens", 0)
    patched_usage.setdefault("cache_creation_input_tokens", 0)
    patched_usage.setdefault("cache_read_input_tokens", 0)
    patched_message["usage"] = patched_usage
    patched_payload["message"] = patched_message
    return patched_payload


class OpencodeCompatHandler(CustomLogger):
    """Compatibility layer for opencode raw DSML/Qwen tool-call output."""

    def __init__(self) -> None:
        _patch_litellm_responses_empty_tools_bridge()
        _patch_litellm_responses_reasoning_text_bridge()
        _patch_litellm_client_disconnect_metadata()
        self._register_input_tokens_route()
        self._register_messages_count_tokens_route()

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

    def _register_messages_count_tokens_route(self) -> None:
        try:
            from fastapi import Depends, Request
            from fastapi.responses import JSONResponse
            from fastapi.routing import APIRoute
            from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
            from litellm.proxy.proxy_server import app

            route_paths = ("/v1/messages/count_tokens", "/messages/count_tokens")
            route_name = "opencode_messages_count_tokens"
            original_endpoints: Dict[str, Any] = {}
            for route in getattr(app, "routes", []):
                if getattr(route, "name", None) == route_name:
                    return
                path = getattr(route, "path", None)
                endpoint = getattr(route, "endpoint", None)
                if path in route_paths and endpoint is not None:
                    original_endpoints[str(path)] = endpoint

            def get_original_endpoint(request: Request) -> Any:
                original = original_endpoints.get(request.url.path)
                if original is not None:
                    return original
                for route in getattr(request.app, "routes", []):
                    if getattr(route, "name", None) == route_name:
                        continue
                    if getattr(route, "path", None) != request.url.path:
                        continue
                    endpoint = getattr(route, "endpoint", None)
                    if endpoint is not None:
                        original_endpoints[request.url.path] = endpoint
                        return endpoint
                return None

            async def opencode_messages_count_tokens(request: Request):
                try:
                    body = await request.body()
                    payload = json.loads(body) if body else {}
                    payload = payload if isinstance(payload, dict) else {}
                except Exception:
                    payload = {}
                estimated_tokens = _estimate_anthropic_messages_tokens(payload)
                if estimated_tokens >= COUNT_TOKENS_NATIVE_MAX_ESTIMATE:
                    return JSONResponse(content={"input_tokens": estimated_tokens})

                original = get_original_endpoint(request)
                if original is not None:
                    try:
                        native_response = await original(request=request)
                        native_tokens = _extract_input_tokens(native_response)
                        if native_tokens and native_tokens > 0:
                            return JSONResponse(content={"input_tokens": native_tokens})
                    except Exception as exc:
                        log.warning("native messages count_tokens failed; using estimate: %s", exc)

                return JSONResponse(content={"input_tokens": estimated_tokens})

            for route_path in reversed(route_paths):
                route = APIRoute(
                    path=route_path,
                    endpoint=opencode_messages_count_tokens,
                    methods=["POST"],
                    name=route_name,
                    dependencies=[Depends(user_api_key_auth)],
                )
                app.router.routes.insert(0, route)

            log.info("registered opencode compatibility routes %s", ", ".join(route_paths))
        except Exception as exc:
            log.warning("failed to register messages count_tokens compatibility route: %s", exc)

    async def async_pre_call_hook(self, user_api_key_dict: Any, cache: Any, data: dict, call_type: str):
        _sanitize_request_tools(data, call_type)
        if call_type == "anthropic_messages":
            thinking = data.get("thinking")
            if isinstance(thinking, dict) and thinking.get("type") == "enabled":
                budget = thinking.get("budget_tokens")
                if isinstance(budget, int) and budget >= 10000:
                    data["reasoning_effort"] = "high"
                elif isinstance(budget, int) and budget >= 5000:
                    data["reasoning_effort"] = "medium"
                else:
                    data["reasoning_effort"] = "low"
                data.pop("thinking", None)
            if data.get("stream") is None:
                data["stream"] = True
        if call_type in ("responses", "aresponses"):
            _disable_responses_reasoning_merge(data)
            _sanitize_response_input_history(data)
            _sanitize_malformed_function_call_history(data)

        if call_type not in ("completion", "acompletion", "chat_completion", "anthropic_messages"):
            return data

        _sanitize_chat_internal_artifact_history(data)
        _normalize_assistant_messages(data.get("messages"))
        request_probe = dict(data)
        request_probe["call_type"] = call_type
        if _is_stop_hook_json_evaluator(request_probe):
            request_context = _request_context(request_probe)
            session_key = _stop_hook_session_key(request_probe, request_context)
            _STOP_HOOK_REQUEST_STARTED_AT[_stop_hook_request_key(session_key)] = time.time()
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
        request_context = _request_context(request_data)
        if _is_messages_stream(request_data):
            async for chunk in self._convert_anthropic_messages_stream(
                response,
                stop_after_first_native_tool=_stop_after_first_native_tool(request_data),
                request_context=request_context,
                request_data=request_data,
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
        raw_think = _raw_think_state()
        last_id = "chatcmpl-opencode-compat"
        last_model = request_data.get("model", "unknown") if request_data else "unknown"
        last_created = int(time.time())
        tool_call_state: Dict[Any, Dict[str, str]] = {}

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
            _patch_tool_call_ids(delta, tool_call_state)
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
                chunk_text += _strip_raw_think_delta(raw_chunk_text, raw_think)

            if not chunk_text:
                continue

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
                idx = find_raw_tool_start(buffer)
                if idx > 0:
                    for item in pending:
                        yield _make_content_chunk(last_id, last_model, last_created, item)
                    if unflushed_text:
                        yield _make_content_chunk(last_id, last_model, last_created, unflushed_text)
                pending.clear()
                unflushed_text = ""
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

        tail = _flush_raw_think_tail(raw_think)
        if tail:
            unflushed_text += tail
        fallback = _hidden_thinking_final_fallback(raw_think, pending, unflushed_text)
        if fallback:
            unflushed_text += fallback
        _raise_empty_unclosed_raw_think(raw_think, request_context, pending, unflushed_text, False)

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
        request_data: Optional[dict] = None,
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
        saw_message_start = False
        synthetic_stop_sent = False
        stop_hook_json_evaluator = _is_stop_hook_json_evaluator(request_data)
        stop_hook_session_key = _stop_hook_session_key(request_data, request_context)
        stop_hook_visible_text = False
        stop_hook_text_buffer = ""
        stop_hook_started_at = time.time()
        if stop_hook_json_evaluator:
            started_key = _stop_hook_request_key(stop_hook_session_key)
            stop_hook_started_at = _STOP_HOOK_REQUEST_STARTED_AT.pop(started_key, stop_hook_started_at)
        openai_sse_mode = False

        keepalive_seconds = STOP_HOOK_KEEPALIVE_SECONDS if stop_hook_json_evaluator else MESSAGES_STREAM_KEEPALIVE_SECONDS

        force_keepalive_at = None
        if stop_hook_json_evaluator:
            force_keepalive_at = stop_hook_started_at + STOP_HOOK_JSON_FALLBACK_SECONDS

        async for chunk in _iter_with_keepalive(
            response,
            request_context,
            keepalive_seconds=keepalive_seconds,
            force_keepalive_at=force_keepalive_at,
        ):
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
                    if openai_sse_mode and "[DONE]" in raw_event:
                        if not saw_message_stop and not synthetic_stop_sent:
                            for index in sorted(open_content_blocks):
                                yield _sse("content_block_stop", {"type": "content_block_stop", "index": index}, chunk)
                            if not saw_stop_message_delta:
                                yield _sse(
                                    "message_delta",
                                    {
                                        "type": "message_delta",
                                        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                                        "usage": {"output_tokens": 0},
                                    },
                                    chunk,
                                )
                            yield _sse("message_stop", {"type": "message_stop"}, chunk)
                            synthetic_stop_sent = True
                        continue
                    if not passthrough_blocked:
                        yield _encode_like(raw_event + "\n\n", chunk)
                    if (
                        stop_hook_json_evaluator
                        and not stop_hook_visible_text
                        and time.time() - stop_hook_started_at >= STOP_HOOK_JSON_FALLBACK_SECONDS
                        and _stop_hook_json_fallback_available(stop_hook_session_key, request_context)
                    ):
                        for event in _messages_text_end_turn_events(
                            _stop_hook_json_fallback_text(),
                            text_block_index,
                            chunk,
                            start_block=not saw_content_block,
                        ):
                            yield event
                        _record_stop_hook_json_fallback(
                            stop_hook_session_key,
                            request_context,
                            "empty-stream",
                        )
                        log.warning("synthesized Stop hook JSON fallback after empty stream context=%s", request_context)
                        return
                    continue

                if event_name in (None, "") and _choice(payload) is not None:
                    openai_sse_mode = True
                    passthrough_blocked = True
                    if not saw_message_start:
                        yield _sse(
                            "message_start",
                            _message_start_with_estimated_usage(
                                {
                                    "type": "message_start",
                                    "message": {
                                        "id": _chunk_id(payload, "msg_opencode_compat"),
                                        "type": "message",
                                        "role": "assistant",
                                        "model": _chunk_model(payload, "unknown"),
                                        "content": [],
                                        "stop_reason": None,
                                        "stop_sequence": None,
                                        "usage": {"input_tokens": 0, "output_tokens": 0},
                                    },
                                },
                                request_data,
                            ),
                            chunk,
                        )
                        saw_message_start = True

                    delta = _delta(payload)
                    complete_openai_tool = _first_complete_openai_tool_call(openai_tool_state, delta)
                    if complete_openai_tool:
                        for index in sorted(open_content_blocks):
                            yield _sse("content_block_stop", {"type": "content_block_stop", "index": index}, chunk)
                        tool_index = max(open_content_blocks | {text_block_index}) + 1 if saw_content_block else 0
                        for event in _messages_tool_use_events([complete_openai_tool], tool_index, chunk):
                            yield event
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
                        synthetic_stop_sent = True
                        log.info("synthesized messages native tool stop from OpenAI SSE context=%s", request_context)
                        return

                    chunk_text = _content_from_delta(delta) or _reasoning_from_delta(delta)
                    if chunk_text:
                        if stop_hook_json_evaluator:
                            stop_hook_text_buffer += chunk_text
                            if _is_valid_stop_hook_json_text(stop_hook_text_buffer):
                                _record_stop_hook_valid_json(stop_hook_session_key, request_context)
                        if not saw_content_block:
                            yield _sse(
                                "content_block_start",
                                {
                                    "type": "content_block_start",
                                    "index": text_block_index,
                                    "content_block": {"type": "text", "text": ""},
                                },
                                chunk,
                            )
                            saw_content_block = True
                            open_content_blocks.add(text_block_index)
                        async for item in self._handle_messages_text_delta(
                            chunk_text,
                            text_block_index,
                            chunk,
                            "text_delta",
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

                    choice = _choice(payload)
                    finish_reason = _get(choice, "finish_reason", None)
                    if finish_reason:
                        tail = _flush_raw_think_tail(raw_think)
                        if tail:
                            unflushed_text += tail
                        fallback = _hidden_thinking_final_fallback(raw_think, pending, unflushed_text)
                        if fallback:
                            yield _messages_text_delta(fallback, text_block_index, chunk, "text_delta")
                        _raise_empty_unclosed_raw_think(
                            raw_think,
                            request_context,
                            pending,
                            unflushed_text,
                            any(bool(entry.get("name")) for entry in openai_tool_state.values()),
                        )
                        for item in pending:
                            yield _messages_text_delta(item, text_block_index, chunk, text_delta_type)
                        pending = []
                        if unflushed_text:
                            yield _messages_text_delta(unflushed_text, text_block_index, chunk, text_delta_type)
                            unflushed_text = ""
                        for index in sorted(open_content_blocks):
                            yield _sse("content_block_stop", {"type": "content_block_stop", "index": index}, chunk)
                        open_content_blocks.clear()
                        stop_reason = "tool_use" if str(finish_reason) == "tool_calls" else "end_turn"
                        yield _sse(
                            "message_delta",
                            {
                                "type": "message_delta",
                                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                                "usage": {"output_tokens": 0},
                            },
                            chunk,
                        )
                        yield _sse("message_stop", {"type": "message_stop"}, chunk)
                        saw_stop_message_delta = True
                        saw_message_stop = True
                        synthetic_stop_sent = True
                        return
                    continue

                if event_name == "content_block_delta":
                    delta = payload.get("delta") or {}
                    delta_type = str(delta.get("type") or "")
                    delta_field = "thinking" if delta_type == "thinking_delta" else "text"
                    if delta_type in {"text_delta", "thinking_delta"} and isinstance(delta.get(delta_field), str):
                        text_block_index = _event_index(payload, text_block_index)
                        text_delta_type = delta_type
                        if stop_hook_json_evaluator and delta_type == "thinking_delta":
                            if raw_think.get("started_at") is None:
                                raw_think["started_at"] = time.time()
                            _record_raw_think_suppressed(delta[delta_field], raw_think)
                            if (
                                not stop_hook_visible_text
                                and time.time() - stop_hook_started_at >= STOP_HOOK_JSON_FALLBACK_SECONDS
                                and _stop_hook_json_fallback_available(stop_hook_session_key, request_context)
                            ):
                                for event in _messages_text_end_turn_events(
                                    _stop_hook_json_fallback_text(),
                                    text_block_index,
                                    chunk,
                                    start_block=not saw_content_block,
                                ):
                                    yield event
                                _record_stop_hook_json_fallback(
                                    stop_hook_session_key,
                                    request_context,
                                    "reasoning-only",
                                )
                                log.warning(
                                    "synthesized Stop hook JSON fallback after reasoning-only stream context=%s chars=%s",
                                    request_context,
                                    raw_think.get("suppressed_chars") or 0,
                                )
                                return
                            continue
                        if stop_hook_json_evaluator and delta_type == "text_delta" and delta[delta_field].strip():
                            stop_hook_visible_text = True
                            stop_hook_text_buffer += delta[delta_field]
                            if _is_valid_stop_hook_json_text(stop_hook_text_buffer):
                                _record_stop_hook_valid_json(stop_hook_session_key, request_context)
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
                        if stop_hook_json_evaluator and unflushed_text.strip():
                            stop_hook_visible_text = True
                        yield _messages_text_delta(unflushed_text, text_block_index, chunk, text_delta_type)
                        unflushed_text = ""
                    text_buffer = ""

                if event_name == "message_start":
                    saw_message_start = True
                    yield _sse(
                        event_name,
                        _message_start_with_estimated_usage(payload, request_data),
                        chunk,
                    )
                else:
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
            visible_prefix = text_buffer[:idx] if idx > 0 else ""
            if idx > 0:
                yield _messages_text_delta(visible_prefix, text_block_index, original_for_output, text_delta_type)
            if idx < len(text_buffer):
                log.warning(
                    "suppressing incomplete messages raw tool block context=%s preview=%r",
                    request_context,
                    text_buffer[idx:idx + 800].replace("\n", "\\n"),
                )
                if (
                    stop_hook_json_evaluator
                    and not synthetic_stop_sent
                    and _stop_hook_json_fallback_available(stop_hook_session_key, request_context)
                ):
                    for event in _messages_text_end_turn_events(
                        _stop_hook_json_fallback_text(),
                        text_block_index,
                        original_for_output,
                        start_block=not saw_content_block,
                    ):
                        yield event
                    _record_stop_hook_json_fallback(
                        stop_hook_session_key,
                        request_context,
                        "incomplete-raw-tool",
                    )
                    log.warning(
                        "synthesized Stop hook JSON fallback after incomplete raw tool block context=%s",
                        request_context,
                    )
                    return
                if not any(bool(item) for item in pending) and not unflushed_text and not visible_prefix.strip():
                    for event in _messages_text_end_turn_events(
                        _incomplete_raw_tool_fallback_text(),
                        text_block_index,
                        original_for_output,
                        start_block=not saw_content_block,
                    ):
                        yield event
                    log.warning(
                        "synthesized malformed fallback after incomplete raw tool block context=%s",
                        request_context,
                    )
                    return
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

        if (
            stop_hook_json_evaluator
            and not stop_hook_visible_text
            and not synthetic_stop_sent
            and _stop_hook_json_fallback_available(stop_hook_session_key, request_context)
        ):
            for event in _messages_text_end_turn_events(
                _stop_hook_json_fallback_text(),
                text_block_index,
                original_for_output,
                start_block=not saw_content_block,
            ):
                yield event
            _record_stop_hook_json_fallback(
                stop_hook_session_key,
                request_context,
                "stream-end",
            )
            log.warning(
                "synthesized Stop hook JSON fallback at stream end context=%s chars=%s",
                request_context,
                raw_think.get("suppressed_chars") or 0,
            )
            return

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

        if delta_type == "thinking_delta":
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
        if dsml_mode:
            safe_text = text
            revealed_hidden_delta = False
        else:
            # Raw tool parsing must only see visible assistant content. If a model emits
            # "<think>\n<tool_call>" without closing the thought, the tool prefix should
            # remain suppressed with the hidden text instead of leaking a bare <think>.
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
                for p in parsed:
                    pf = p.get("function", {})
                    log.info("parsed_raw_tool name=%r args_preview=%s", pf.get("name"), str(pf.get("arguments", ""))[:200])
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
            idx = find_raw_tool_start(text_buffer)
            if idx > 0:
                for item in pending:
                    yield _messages_text_delta(item, text_block_index, original, delta_type)
                if unflushed_text:
                    yield _messages_text_delta(unflushed_text, text_block_index, original, delta_type)
            pending = []
            unflushed_text = ""
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
