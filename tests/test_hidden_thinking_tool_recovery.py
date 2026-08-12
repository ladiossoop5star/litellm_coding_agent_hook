import asyncio
import json
import time
import unittest
from unittest.mock import patch

from opencode_compat_hook.hook import (
    OpencodeCompatHandler,
    _coerce_value_for_schema,
    _deployment_api_base,
    _estimate_count_tokens,
    _hoist_system_chat_messages,
    _hoist_system_responses_input,
    _iter_with_keepalive,
    _is_stop_hook_json_evaluator,
    _raw_think_state,
    _remove_stop_hook_structured_output,
    _request_tool_schemas,
    _server_info_lacks_grammar,
    _server_info_url,
    _truncate_stop_hook_history,
)

from opencode_compat_hook.hook import (
    STOP_HOOK_HISTORY_HEAD_MAX_TOKENS,
    STOP_HOOK_HISTORY_TAIL_MAX_TOKENS,
)


class CloseableBlockingStream:
    def __init__(self, first_chunk):
        self.first_chunk = first_chunk
        self.reads = 0
        self.pending_read_started = asyncio.Event()
        self.pending_read_cancelled = asyncio.Event()
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        self.reads += 1
        if self.reads == 1:
            return self.first_chunk
        self.pending_read_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.pending_read_cancelled.set()
            raise

    async def aclose(self):
        self.closed = True


BASH_SCHEMA = {
    "type": "object",
    "properties": {
        "command": {"type": "string"},
        "description": {"type": "string"},
    },
    "required": ["command", "description"],
    "additionalProperties": False,
}

READ_SCHEMA = {
    "type": "object",
    "properties": {
        "file_path": {"type": "string"},
        "offset": {"type": "integer"},
        "limit": {"type": "integer"},
    },
    "required": ["file_path"],
}

GREP_SCHEMA = {
    "type": "object",
    "properties": {
        "pattern": {"type": "string"},
        "head_limit": {"type": "number"},
        "multiline": {"type": "boolean"},
    },
    "required": ["pattern"],
}


def state():
    raw_think = _raw_think_state()
    raw_think["tool_schemas"] = {"Bash": BASH_SCHEMA}
    return {
        "text_buffer": "",
        "unflushed_text": "",
        "pending": [],
        "dsml_mode": False,
        "raw_think": raw_think,
    }


def state_with_tools(tool_schemas):
    current = state()
    current["raw_think"]["tool_schemas"] = tool_schemas
    return current


async def feed(handler, chunks, current_state):
    emitted = []
    for chunk in chunks:
        async for item in handler._handle_messages_text_delta(
            chunk, 0, b"", "text_delta", current_state
        ):
            if isinstance(item, dict) and item.get("_state"):
                current_state = {
                    "text_buffer": item["text_buffer"],
                    "unflushed_text": item["unflushed_text"],
                    "pending": item["pending"],
                    "dsml_mode": item["dsml_mode"],
                    "raw_think": item["raw_think"],
                }
            else:
                emitted.append(item.decode() if isinstance(item, bytes) else str(item))
    return "".join(emitted), current_state


def stop_hook_request():
    return {
        "call_type": "anthropic_messages",
        "stream": True,
        "messages": [
            {
                "role": "user",
                "content": (
                    '{"hook_event_name":"Stop"} Check the stopping condition '
                    "and return JSON. ARGUMENTS follow."
                ),
            }
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "schema": {
                    "type": "object",
                    "required": ["ok", "reason", "impossible"],
                }
            },
        },
    }


async def anthropic_text_stream(chunks):
    yield (
        'event: message_start\ndata: {"type":"message_start","message":'
        '{"id":"msg_test","type":"message","role":"assistant","model":"test",'
        '"content":[],"stop_reason":null,"usage":{"input_tokens":1,"output_tokens":0}}}\n\n'
    ).encode()
    yield (
        'event: content_block_start\ndata: {"type":"content_block_start","index":0,'
        '"content_block":{"type":"text","text":""}}\n\n'
    ).encode()
    for chunk in chunks:
        yield (
            "event: content_block_delta\ndata: "
            + '{"type":"content_block_delta","index":0,"delta":'
            + '{"type":"text_delta","text":'
            + json.dumps(chunk)
            + "}}\n\n"
        ).encode()
    yield (
        'event: message_delta\ndata: {"type":"message_delta","delta":'
        '{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":1}}\n\n'
    ).encode()
    yield b'event: message_stop\ndata: {"type":"message_stop"}\n\n'


async def anthropic_thinking_then_text_stream(thinking_chunks, text_chunks):
    yield (
        'event: message_start\ndata: {"type":"message_start","message":'
        '{"id":"msg_test","type":"message","role":"assistant","model":"test",'
        '"content":[],"stop_reason":null,"usage":{"input_tokens":1,"output_tokens":0}}}\n\n'
    ).encode()
    yield (
        'event: content_block_start\ndata: {"type":"content_block_start","index":0,'
        '"content_block":{"type":"thinking","thinking":""}}\n\n'
    ).encode()
    for chunk in thinking_chunks:
        yield (
            "event: content_block_delta\ndata: "
            + '{"type":"content_block_delta","index":0,"delta":'
            + '{"type":"thinking_delta","thinking":'
            + json.dumps(chunk)
            + "}}\n\n"
        ).encode()
    yield b'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\n'
    yield (
        'event: content_block_start\ndata: {"type":"content_block_start","index":1,'
        '"content_block":{"type":"text","text":""}}\n\n'
    ).encode()
    for chunk in text_chunks:
        yield (
            "event: content_block_delta\ndata: "
            + '{"type":"content_block_delta","index":1,"delta":'
            + '{"type":"text_delta","text":'
            + json.dumps(chunk)
            + "}}\n\n"
        ).encode()
    yield (
        'event: message_delta\ndata: {"type":"message_delta","delta":'
        '{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":1}}\n\n'
    ).encode()
    yield b'event: message_stop\ndata: {"type":"message_stop"}\n\n'


async def anthropic_text_with_block_stop_stream(chunks):
    yield (
        'event: message_start\ndata: {"type":"message_start","message":'
        '{"id":"msg_test","type":"message","role":"assistant","model":"test",'
        '"content":[],"stop_reason":null,"usage":{"input_tokens":1,"output_tokens":0}}}\n\n'
    ).encode()
    yield (
        'event: content_block_start\ndata: {"type":"content_block_start","index":0,'
        '"content_block":{"type":"text","text":""}}\n\n'
    ).encode()
    for chunk in chunks:
        yield (
            "event: content_block_delta\ndata: "
            + '{"type":"content_block_delta","index":0,"delta":'
            + '{"type":"text_delta","text":'
            + json.dumps(chunk)
            + "}}\n\n"
        ).encode()
    yield b'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\n'
    yield (
        'event: message_delta\ndata: {"type":"message_delta","delta":'
        '{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":1}}\n\n'
    ).encode()
    yield b'event: message_stop\ndata: {"type":"message_stop"}\n\n'


async def anthropic_text_stream_stop_only(chunks):
    yield (
        'event: message_start\ndata: {"type":"message_start","message":'
        '{"id":"msg_test","type":"message","role":"assistant","model":"test",'
        '"content":[],"stop_reason":null,"usage":{"input_tokens":1,"output_tokens":0}}}\n\n'
    ).encode()
    yield (
        'event: content_block_start\ndata: {"type":"content_block_start","index":0,'
        '"content_block":{"type":"text","text":""}}\n\n'
    ).encode()
    for chunk in chunks:
        yield (
            "event: content_block_delta\ndata: "
            + '{"type":"content_block_delta","index":0,"delta":'
            + '{"type":"text_delta","text":'
            + json.dumps(chunk)
            + "}}\n\n"
        ).encode()
    yield b'event: message_stop\ndata: {"type":"message_stop"}\n\n'


async def anthropic_text_in_content_block_start_stream(text):
    yield (
        'event: message_start\ndata: {"type":"message_start","message":'
        '{"id":"msg_test","type":"message","role":"assistant","model":"test",'
        '"content":[],"stop_reason":null,"usage":{"input_tokens":1,"output_tokens":0}}}\n\n'
    ).encode()
    yield (
        "event: content_block_start\ndata: "
        + '{"type":"content_block_start","index":0,"content_block":'
        + '{"type":"text","text":'
        + json.dumps(text)
        + "}}\n\n"
    ).encode()
    yield b'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\n'
    yield (
        'event: message_delta\ndata: {"type":"message_delta","delta":'
        '{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":1}}\n\n'
    ).encode()
    yield b'event: message_stop\ndata: {"type":"message_stop"}\n\n'


async def openai_text_stream(chunks, finish_reason="stop"):
    yield (
        'data: {"id":"chatcmpl_test","object":"chat.completion.chunk","model":"test",'
        '"choices":[{"index":0,"delta":{"role":"assistant","content":""},'
        '"finish_reason":null}]}\n\n'
    ).encode()
    for chunk in chunks:
        yield (
            'data: {"id":"chatcmpl_test","object":"chat.completion.chunk","model":"test",'
            '"choices":[{"index":0,"delta":{"content":'
            + json.dumps(chunk)
            + '},"finish_reason":null}]}\n\n'
        ).encode()
    yield (
        'data: {"id":"chatcmpl_test","object":"chat.completion.chunk","model":"test",'
        '"choices":[{"index":0,"delta":{},"finish_reason":"'
        + finish_reason
        + '"}]}\n\n'
    ).encode()
    yield b"data: [DONE]\n\n"


async def openai_stream_no_finish_reason(chunks, include_done=True):
    yield (
        'data: {"id":"chatcmpl_test","object":"chat.completion.chunk","model":"test",'
        '"choices":[{"index":0,"delta":{"role":"assistant","content":""},'
        '"finish_reason":null}]}\n\n'
    ).encode()
    for chunk in chunks:
        yield (
            'data: {"id":"chatcmpl_test","object":"chat.completion.chunk","model":"test",'
            '"choices":[{"index":0,"delta":{"content":'
            + json.dumps(chunk)
            + '},"finish_reason":null}]}\n\n'
        ).encode()
    if include_done:
        yield b"data: [DONE]\n\n"


async def anthropic_stream_with_transparent_retry():
    yield (
        'event: message_start\ndata: {"type":"message_start","message":'
        '{"id":"msg_first","type":"message","role":"assistant","model":"test",'
        '"content":[],"stop_reason":null,"usage":{"input_tokens":1,"output_tokens":0}}}\n\n'
    ).encode()
    yield (
        'event: content_block_start\ndata: {"type":"content_block_start","index":0,'
        '"content_block":{"type":"text","text":""}}\n\n'
    ).encode()
    yield (
        'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,'
        '"delta":{"type":"text_delta","text":"partial first attempt"}}\n\n'
    ).encode()
    # LiteLLM can transparently retry an upstream request after the first attempt
    # has already emitted SSE. A second message_start in the same HTTP response is
    # invalid Anthropic Messages protocol and must not be forwarded.
    yield (
        'event: message_start\ndata: {"type":"message_start","message":'
        '{"id":"msg_retry","type":"message","role":"assistant","model":"test",'
        '"content":[],"stop_reason":null,"usage":{"input_tokens":1,"output_tokens":0}}}\n\n'
    ).encode()
    yield (
        'event: content_block_start\ndata: {"type":"content_block_start","index":0,'
        '"content_block":{"type":"tool_use","id":"tool_retry","name":"Bash","input":{}}}\n\n'
    ).encode()


def emitted_text(rendered):
    text = []
    for line in rendered.splitlines():
        if not line.startswith("data: "):
            continue
        payload = json.loads(line[6:])
        delta = payload.get("delta") or {}
        if delta.get("type") == "text_delta":
            text.append(delta.get("text") or "")
    return "".join(text)


def content_block_start_indexes(rendered):
    indexes = []
    for line in rendered.splitlines():
        if not line.startswith("data: "):
            continue
        payload = json.loads(line[6:])
        if payload.get("type") == "content_block_start":
            indexes.append(payload.get("index"))
    return indexes


class HiddenThinkingToolRecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.handler = object.__new__(OpencodeCompatHandler)

    async def test_keepalive_closes_only_its_request_stream_on_early_exit(self):
        cancelled_stream = CloseableBlockingStream(b"first")
        unaffected_stream = CloseableBlockingStream(b"other")
        cancelled_generator = _iter_with_keepalive(
            cancelled_stream,
            request_context="cancelled-request",
        )
        unaffected_generator = _iter_with_keepalive(
            unaffected_stream,
            request_context="unaffected-request",
        )

        self.assertEqual(await cancelled_generator.__anext__(), b"first")
        self.assertEqual(await unaffected_generator.__anext__(), b"other")
        cancelled_next = asyncio.create_task(cancelled_generator.__anext__())
        await asyncio.wait_for(cancelled_stream.pending_read_started.wait(), timeout=1.0)

        cancelled_next.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await cancelled_next

        self.assertTrue(cancelled_stream.pending_read_cancelled.is_set())
        self.assertTrue(cancelled_stream.closed)
        self.assertFalse(unaffected_stream.pending_read_cancelled.is_set())
        self.assertFalse(unaffected_stream.closed)

        await asyncio.wait_for(unaffected_generator.aclose(), timeout=1.0)
        self.assertTrue(unaffected_stream.closed)

    async def test_messages_hook_closes_provider_stream_when_client_stops(self):
        first_event = (
            'event: message_start\ndata: {"type":"message_start","message":'
            '{"id":"msg_cancel","type":"message","role":"assistant",'
            '"model":"test","content":[],"stop_reason":null,'
            '"usage":{"input_tokens":1,"output_tokens":0}}}\n\n'
        ).encode()
        provider_stream = CloseableBlockingStream(first_event)
        output_stream = self.handler.async_post_call_streaming_iterator_hook(
            None,
            provider_stream,
            {"call_type": "anthropic_messages", "stream": True},
        )

        first_output = await asyncio.wait_for(output_stream.__anext__(), timeout=1.0)
        self.assertIn(b"event: message_start", first_output)

        await asyncio.wait_for(output_stream.aclose(), timeout=1.0)

        self.assertTrue(provider_stream.closed)

    async def test_complete_tool_call_implicitly_closes_unclosed_thinking(self):
        output, current = await feed(
            self.handler,
            [
                "<think>Good, now rebuild.\n<tool_",
                "call>\n<function=Bash>\n<parameter=command>\nmake -j128\n</parameter>\n"
                "<parameter=description>\nBuild firmware\n</parameter>\n</function>\n</tool_call>\n",
            ],
            state(),
        )

        self.assertNotIn("<tool_call>", output)
        self.assertIn('"type": "tool_use"', output)
        self.assertIn('"name": "Bash"', output)
        self.assertIn('"stop_reason": "tool_use"', output)
        self.assertFalse(current["raw_think"]["in_think"])

    async def test_revealed_thinking_still_recovers_tool_without_xml_leak(self):
        current = state()
        _, current = await feed(self.handler, ["<think>long reasoning"], current)
        current["raw_think"]["started_at"] = time.time() - 31

        output, _ = await feed(
            self.handler,
            [
                " continues\n<tool_call>\n<function=Bash>\n"
                "<parameter=command>\ngit show 4fbd2a9b\n</parameter>\n"
                "<parameter=description>\nCheck commit\n</parameter>\n"
                "</function>\n</tool_call>\n"
            ],
            current,
        )

        self.assertNotIn("<tool_call>", output)
        self.assertIn('"type": "tool_use"', output)
        self.assertIn("git show 4fbd2a9b", output)

    async def test_unknown_hidden_tool_is_not_executed(self):
        output, _ = await feed(
            self.handler,
            [
                "<think>try this\n<tool_call>\n<function=Unknown>\n"
                "<parameter=value>\n1\n</parameter>\n</function>\n</tool_call>\n"
            ],
            state(),
        )

        self.assertNotIn('"type": "tool_use"', output)
        self.assertIn("model output malformed", output)
        self.assertIn('"stop_reason": "end_turn"', output)

    async def test_schema_invalid_hidden_tool_is_not_executed(self):
        output, _ = await feed(
            self.handler,
            [
                "<think>try this\n<tool_call>\n<function=Bash>\n"
                "<parameter=command>\nmake\n</parameter>\n</function>\n</tool_call>\n"
            ],
            state(),
        )

        self.assertNotIn('"type": "tool_use"', output)
        self.assertIn("model output malformed", output)

    async def test_integer_string_hidden_tool_argument_is_coerced_and_executed(self):
        output, _ = await feed(
            self.handler,
            [
                "<think>read the file\n<tool_call>\n<function=Read>\n"
                "<parameter=file_path>\n/tmp/a.txt\n</parameter>\n"
                "<parameter=limit>\n120\n</parameter>\n</function>\n</tool_call>\n"
            ],
            state_with_tools({"Read": READ_SCHEMA}),
        )

        self.assertIn('"type": "tool_use"', output)
        self.assertIn('"name": "Read"', output)
        self.assertIn('\\"limit\\": 120', output)
        self.assertNotIn("model output malformed", output)

    async def test_normal_tool_block_arguments_are_coerced(self):
        output, _ = await feed(
            self.handler,
            [
                "<think>plan</think>\n<tool_call>\n<function=Grep>\n"
                "<parameter=pattern>\nfoo\n</parameter>\n"
                "<parameter=head_limit>\n12.5\n</parameter>\n"
                "<parameter=multiline>\ntrue\n</parameter>\n"
                "</function>\n</tool_call>\n"
            ],
            state_with_tools({"Grep": GREP_SCHEMA}),
        )

        self.assertIn('"type": "tool_use"', output)
        self.assertIn('\\"head_limit\\": 12.5', output)
        self.assertIn('\\"multiline\\": true', output)

    async def test_non_numeric_string_hidden_tool_argument_is_still_rejected(self):
        output, _ = await feed(
            self.handler,
            [
                "<think>try this\n<tool_call>\n<function=Read>\n"
                "<parameter=file_path>\n/tmp/a.txt\n</parameter>\n"
                "<parameter=limit>\nabc\n</parameter>\n</function>\n</tool_call>\n"
            ],
            state_with_tools({"Read": READ_SCHEMA}),
        )

        self.assertNotIn('"type": "tool_use"', output)
        self.assertIn("model output malformed", output)

    def test_coerce_value_for_schema_scalars(self):
        self.assertEqual(_coerce_value_for_schema("120", {"type": "integer"}), 120)
        self.assertEqual(_coerce_value_for_schema(" -3 ", {"type": "integer"}), -3)
        self.assertEqual(_coerce_value_for_schema("12.5", {"type": "number"}), 12.5)
        self.assertEqual(_coerce_value_for_schema("7", {"type": "number"}), 7)
        self.assertIs(_coerce_value_for_schema("true", {"type": "boolean"}), True)
        self.assertIs(_coerce_value_for_schema("FALSE", {"type": "boolean"}), False)
        self.assertEqual(_coerce_value_for_schema("abc", {"type": "integer"}), "abc")
        self.assertEqual(_coerce_value_for_schema("120", {"type": "string"}), "120")
        self.assertEqual(_coerce_value_for_schema(120, {"type": "integer"}), 120)
        self.assertEqual(_coerce_value_for_schema("120", {"type": ["integer", "null"]}), 120)
        self.assertEqual(
            _coerce_value_for_schema({"limit": "5"}, {"properties": {"limit": {"type": "integer"}}}),
            {"limit": 5},
        )
        self.assertEqual(
            _coerce_value_for_schema(["1", "2"], {"type": "array", "items": {"type": "integer"}}),
            [1, 2],
        )
        self.assertEqual(
            _coerce_value_for_schema("120", {"anyOf": [{"type": "integer"}, {"type": "string"}]}),
            120,
        )

    def test_hoist_system_chat_messages_merges_into_leading_system(self):
        messages = [
            {"role": "system", "content": "base"},
            {"role": "user", "content": "hi"},
            {"role": "system", "content": "extra"},
            {"role": "developer", "content": [{"type": "text", "text": "dev note"}]},
            {"role": "user", "content": "go"},
        ]

        folded = _hoist_system_chat_messages(messages)

        self.assertEqual(folded, 3)
        self.assertEqual(
            messages,
            [
                {"role": "system", "content": "base\n\nextra\n\ndev note"},
                {"role": "user", "content": "hi"},
                {"role": "user", "content": "go"},
            ],
        )

    def test_hoist_system_chat_messages_noop_when_well_formed(self):
        messages = [
            {"role": "system", "content": "base"},
            {"role": "user", "content": "hi"},
        ]

        self.assertEqual(_hoist_system_chat_messages(messages), 0)
        self.assertEqual(messages[0]["content"], "base")

    def test_hoist_system_responses_input_merges_instructions(self):
        data = {
            "instructions": "base instructions",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hi"}],
                },
                {
                    "type": "message",
                    "role": "system",
                    "content": [{"type": "input_text", "text": "mid system"}],
                },
                {"type": "function_call", "name": "f", "call_id": "c1", "arguments": "{}"},
            ],
        }

        folded = _hoist_system_responses_input(data)

        self.assertEqual(folded, 1)
        self.assertEqual(data["instructions"], "base instructions\n\nmid system")
        self.assertEqual(len(data["input"]), 2)
        self.assertEqual(data["input"][0]["role"], "user")
        self.assertEqual(data["input"][1]["type"], "function_call")

    def test_hoist_system_responses_input_noop_without_instruction_items(self):
        data = {
            "instructions": "base instructions",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hi"}],
                }
            ],
        }

        self.assertEqual(_hoist_system_responses_input(data), 0)
        self.assertEqual(data["instructions"], "base instructions")
        self.assertEqual(len(data["input"]), 1)

    async def test_stop_hook_fallback_block_index_rebased_to_zero(self):
        output = []
        async for item in self.handler._convert_anthropic_messages_stream(
            anthropic_thinking_then_text_stream(
                ["checking the situation"],
                ["I looked at the logs and the answer is prose."],
            ),
            request_context="test-fallback-reindex-%s" % time.time(),
            request_data=stop_hook_request(),
        ):
            output.append(item.decode() if isinstance(item, bytes) else str(item))

        rendered = "".join(output)
        indexes = content_block_start_indexes(rendered)
        self.assertTrue(indexes)
        self.assertEqual(indexes, [0] * len(indexes))
        self.assertIn("No usable Stop hook JSON", rendered)

    async def test_stop_hook_valid_json_block_index_rebased_to_zero(self):
        valid_json = '{"ok":true,"reason":"all done","impossible":false}'
        output = []
        with patch("opencode_compat_hook.hook._record_stop_hook_valid_json"):
            async for item in self.handler._convert_anthropic_messages_stream(
                anthropic_thinking_then_text_stream(
                    ["thinking about it"],
                    [valid_json[:15], valid_json[15:]],
                ),
                request_context="test-valid-reindex-%s" % time.time(),
                request_data=stop_hook_request(),
            ):
                output.append(item.decode() if isinstance(item, bytes) else str(item))

        rendered = "".join(output)
        self.assertEqual(content_block_start_indexes(rendered), [0])
        self.assertEqual(emitted_text(rendered), valid_json)

    async def test_explicitly_closed_thinking_keeps_normal_tool_conversion(self):
        output, _ = await feed(
            self.handler,
            [
                "<think>plan</think>\n<tool_call>\n<function=Bash>\n"
                "<parameter=command>\nmake\n</parameter>\n"
                "<parameter=description>\nBuild\n</parameter>\n"
                "</function>\n</tool_call>\n"
            ],
            state(),
        )

        self.assertNotIn("<think>", output)
        self.assertNotIn("<tool_call>", output)
        self.assertIn('"type": "tool_use"', output)
        self.assertIn('"stop_reason": "tool_use"', output)

    def test_anthropic_request_tools_are_available_for_recovery_validation(self):
        schemas = _request_tool_schemas(
            {"tools": [{"name": "Bash", "input_schema": BASH_SCHEMA}]}
        )

        self.assertEqual(schemas, {"Bash": BASH_SCHEMA})

    def test_stop_hook_detection_handles_cyclic_request_data(self):
        cyclic_metadata = {}
        cyclic_metadata["self"] = cyclic_metadata
        cyclic_metadata["marker"] = (
            '{"hook_event_name":"Stop"} Check the stopping condition; ARGUMENTS follow.'
        )
        request_data = {
            "call_type": "anthropic_messages",
            "stream": True,
            "metadata": cyclic_metadata,
            "response_format": {
                "json_schema": {
                    "schema": {
                        "required": ["ok", "reason", "impossible"],
                    }
                }
            },
        }
        request_data["cycle"] = request_data

        self.assertTrue(_is_stop_hook_json_evaluator(request_data))

    async def test_stop_hook_request_disables_reasoning_content_merge(self):
        request_data = stop_hook_request()
        request_data["merge_reasoning_content_in_choices"] = True

        result = await self.handler.async_pre_call_hook(
            None,
            None,
            request_data,
            "anthropic_messages",
        )

        self.assertFalse(result["merge_reasoning_content_in_choices"])

    def _build_stop_hook_history(self, count, condition_index=0, content_size=500):
        messages = []
        padding = "x" * content_size
        for index in range(count):
            if index == condition_index:
                content = (
                    '{"hook_event_name":"Stop"} Check the stopping condition '
                    "and return JSON. ARGUMENTS follow."
                )
            else:
                content = f"user message {index} {padding}"
            messages.append(
                {
                    "role": "assistant" if index % 2 == 1 else "user",
                    "content": content,
                }
            )
        return messages

    def test_truncate_stop_hook_history_keeps_head_and_tail(self):
        messages = self._build_stop_hook_history(80, content_size=500)
        data = {"messages": list(messages)}

        dropped = _truncate_stop_hook_history(data)

        self.assertGreater(dropped, 0)
        self.assertEqual(data["messages"][0], messages[0])
        self.assertEqual(data["messages"][-1], messages[-1])
        kept = data["messages"]
        budget = STOP_HOOK_HISTORY_HEAD_MAX_TOKENS + STOP_HOOK_HISTORY_TAIL_MAX_TOKENS
        self.assertLessEqual(sum(_estimate_count_tokens(m) for m in kept), budget + 300)

    def test_truncate_stop_hook_history_drops_only_middle(self):
        messages = self._build_stop_hook_history(80, content_size=500)
        data = {"messages": list(messages)}

        dropped = _truncate_stop_hook_history(data)

        kept = data["messages"]
        self.assertEqual(dropped, len(messages) - len(kept))
        head = next((i for i, m in enumerate(kept) if m != messages[i]), len(kept))
        tail = len(kept) - head
        self.assertLess(head, len(kept))
        self.assertEqual(kept[:head], messages[:head])
        self.assertEqual(kept[head:], messages[len(messages) - tail:])

    def test_truncate_stop_hook_history_keeps_condition_prompt(self):
        messages = self._build_stop_hook_history(80, condition_index=40, content_size=500)
        data = {"messages": list(messages)}

        dropped = _truncate_stop_hook_history(data)

        self.assertGreater(dropped, 0)
        kept_text = " ".join(
            m.get("content", "") if isinstance(m.get("content"), str) else ""
            for m in data["messages"]
        )
        self.assertIn("stopping condition", kept_text)
        self.assertIn("hook_event_name", kept_text)

    def test_truncate_stop_hook_history_keeps_condition_prompt_in_tail(self):
        messages = self._build_stop_hook_history(80, condition_index=79, content_size=500)
        data = {"messages": list(messages)}

        dropped = _truncate_stop_hook_history(data)

        self.assertGreater(dropped, 0)
        self.assertEqual(data["messages"][-1]["content"], messages[79]["content"])

    def test_truncate_stop_hook_history_short_history_untouched(self):
        messages = self._build_stop_hook_history(5, content_size=10)
        data = {"messages": list(messages)}

        dropped = _truncate_stop_hook_history(data)

        self.assertEqual(dropped, 0)
        self.assertEqual(data["messages"], messages)

    async def test_pre_call_hook_truncates_stop_hook_history(self):
        request_data = stop_hook_request()
        request_data["messages"] = self._build_stop_hook_history(80, content_size=500)

        result = await self.handler.async_pre_call_hook(
            None,
            None,
            request_data,
            "anthropic_messages",
        )

        self.assertGreater(len(result["messages"]), 0)
        self.assertLess(len(result["messages"]), 80)
        self.assertFalse(result["merge_reasoning_content_in_choices"])

    def test_stop_hook_deployment_capability_helpers(self):
        self.assertEqual(
            _deployment_api_base({"litellm_params": {"api_base": "http://pgc2:9527/v1/"}}),
            "http://pgc2:9527/v1",
        )
        self.assertEqual(
            _server_info_url("http://pgc2:9527/v1"),
            "http://pgc2:9527/get_server_info",
        )
        self.assertTrue(_server_info_lacks_grammar({"speculative_algorithm": "DFLASH"}))
        self.assertFalse(_server_info_lacks_grammar({"speculative_algorithm": "EAGLE"}))
        self.assertFalse(_server_info_lacks_grammar({"served_model_name": "qwen"}))

    def test_remove_stop_hook_structured_output_preserves_other_output_options(self):
        request_data = stop_hook_request()
        request_data["output_config"] = {
            "verbosity": "low",
            "format": request_data["response_format"],
        }

        self.assertTrue(_remove_stop_hook_structured_output(request_data))
        self.assertNotIn("response_format", request_data)
        self.assertEqual(request_data["output_config"], {"verbosity": "low"})

    async def test_deployment_hook_only_downgrades_incompatible_stop_evaluator(self):
        stop_request = stop_hook_request()
        stop_request["api_base"] = "http://pgc2:9527/v1"
        with patch.object(
            self.handler,
            "_deployment_lacks_stop_hook_grammar",
            return_value=True,
        ):
            result = await self.handler.async_pre_call_deployment_hook(
                stop_request,
                "anthropic_messages",
            )
        self.assertIs(result, stop_request)
        self.assertNotIn("response_format", stop_request)

        ordinary_request = {
            "api_base": "http://pgc2:9527/v1",
            "call_type": "anthropic_messages",
            "stream": True,
            "messages": [{"role": "user", "content": "hello"}],
            "response_format": {"type": "json_schema"},
        }
        with patch.object(
            self.handler,
            "_deployment_lacks_stop_hook_grammar",
            side_effect=AssertionError("ordinary requests must not probe capabilities"),
        ):
            result = await self.handler.async_pre_call_deployment_hook(
                ordinary_request,
                "anthropic_messages",
            )
        self.assertIsNone(result)
        self.assertIn("response_format", ordinary_request)

    async def test_deployment_hook_keeps_schema_for_supported_deployment(self):
        request_data = stop_hook_request()
        request_data["api_base"] = "http://other:9527/v1"
        with patch.object(
            self.handler,
            "_deployment_lacks_stop_hook_grammar",
            return_value=False,
        ):
            result = await self.handler.async_pre_call_deployment_hook(
                request_data,
                "anthropic_messages",
            )
        self.assertIsNone(result)
        self.assertIn("response_format", request_data)

    async def test_transparent_retry_is_ended_before_duplicate_message_start(self):
        output = []
        async for item in self.handler._convert_anthropic_messages_stream(
            anthropic_stream_with_transparent_retry(),
            request_context="test-transparent-retry",
            request_data={"call_type": "anthropic_messages", "stream": True},
        ):
            output.append(item.decode() if isinstance(item, bytes) else str(item))

        rendered = "".join(output)
        self.assertEqual(rendered.count("event: message_start\n"), 1)
        self.assertEqual(rendered.count("event: message_stop\n"), 1)
        self.assertEqual(rendered.count("event: content_block_start\n"), 1)
        self.assertEqual(rendered.count("event: content_block_stop\n"), 1)
        self.assertIn("partial first attempt", emitted_text(rendered))
        self.assertIn('"stop_reason": "end_turn"', rendered)
        self.assertNotIn("msg_retry", rendered)
        self.assertNotIn("tool_retry", rendered)

    async def test_stop_hook_buffers_and_emits_only_complete_valid_json(self):
        valid_json = '{"ok":false,"reason":"work remains","impossible":false}'
        output = []
        with (
            patch(
                "opencode_compat_hook.hook._record_stop_hook_valid_json"
            ) as record_valid,
            patch(
                "opencode_compat_hook.hook.STOP_HOOK_JSON_FALLBACK_SECONDS",
                60.0,
            ),
        ):
            async for item in self.handler._convert_anthropic_messages_stream(
                anthropic_text_stream([valid_json[:12], valid_json[12:]]),
                request_context="test-valid-stop-hook",
                request_data=stop_hook_request(),
            ):
                output.append(item.decode() if isinstance(item, bytes) else str(item))

        rendered = "".join(output)
        self.assertEqual(emitted_text(rendered), valid_json)
        self.assertIn('"stop_reason": "end_turn"', rendered)
        record_valid.assert_called_once()

    async def test_stop_hook_allows_active_reasoning_to_finish_valid_json(self):
        valid_json = '{"ok":true,"reason":"all checks passed","impossible":false}'
        output = []
        with (
            patch(
                "opencode_compat_hook.hook.STOP_HOOK_JSON_FALLBACK_SECONDS",
                0.0,
            ),
            patch(
                "opencode_compat_hook.hook.STOP_HOOK_JSON_ACTIVE_MAX_SECONDS",
                60.0,
            ),
            patch(
                "opencode_compat_hook.hook._record_stop_hook_valid_json"
            ) as record_valid,
        ):
            async for item in self.handler._convert_anthropic_messages_stream(
                anthropic_thinking_then_text_stream(
                    ["Evaluate the evidence first. ", "The condition is satisfied."],
                    [valid_json[:10], valid_json[10:]],
                ),
                request_context="test-reasoning-stop-hook",
                request_data=stop_hook_request(),
            ):
                output.append(item.decode() if isinstance(item, bytes) else str(item))

        rendered = "".join(output)
        self.assertEqual(emitted_text(rendered), valid_json)
        self.assertNotIn("Evaluate the evidence", rendered)
        self.assertIn('"stop_reason": "end_turn"', rendered)
        record_valid.assert_called_once()

    async def test_stop_hook_extracts_typed_json_from_goal_complete_wrapper(self):
        wrapped = (
            "The work is complete.\n<goal-complete>\n"
            '{"ok":true,"reason":"verified","impossible":false}'
            "\n</goal-complete>"
        )
        output = []
        async for item in self.handler._convert_anthropic_messages_stream(
            anthropic_text_stream([wrapped]),
            request_context="test-wrapped-stop-hook",
            request_data=stop_hook_request(),
        ):
            output.append(item.decode() if isinstance(item, bytes) else str(item))

        decision = json.loads(emitted_text("".join(output)))
        self.assertTrue(decision["ok"])
        self.assertEqual(decision["reason"], "verified")
        self.assertFalse(decision["impossible"])

    async def test_stop_hook_reasoning_only_emits_one_terminal_sequence(self):
        output = []
        with patch(
            "opencode_compat_hook.hook._stop_hook_json_fallback_available",
            return_value=True,
        ):
            async for item in self.handler._convert_anthropic_messages_stream(
                anthropic_thinking_then_text_stream(
                    ["The model considered the evidence but omitted its JSON."],
                    [],
                ),
                request_context="test-reasoning-only-stop-hook",
                request_data=stop_hook_request(),
            ):
                output.append(item.decode() if isinstance(item, bytes) else str(item))

        rendered = "".join(output)
        decision = json.loads(emitted_text(rendered))
        self.assertFalse(decision["ok"])
        self.assertEqual(rendered.count("event: message_start\n"), 1)
        self.assertEqual(rendered.count("event: content_block_start\n"), 1)
        self.assertEqual(rendered.count("event: content_block_stop\n"), 1)
        self.assertEqual(rendered.count("event: message_delta\n"), 1)
        self.assertEqual(rendered.count("event: message_stop\n"), 1)

    async def test_stop_hook_accepts_json_in_content_block_start(self):
        valid_json = '{"ok":true,"reason":"verified by test","impossible":false}'
        output = []
        async for item in self.handler._convert_anthropic_messages_stream(
            anthropic_text_in_content_block_start_stream(valid_json),
            request_context="test-start-text-stop-hook",
            request_data=stop_hook_request(),
        ):
            output.append(item.decode() if isinstance(item, bytes) else str(item))

        rendered = "".join(output)
        decision = json.loads(emitted_text(rendered))
        self.assertTrue(decision["ok"])
        self.assertEqual(decision["reason"], "verified by test")
        self.assertEqual(rendered.count("event: message_stop\n"), 1)

    async def test_stop_hook_repairs_json_prefix_lost_at_reasoning_boundary(self):
        merged = (
            "<think>The evidence proves the goal is met.</think>"
            'ok":true,"reason":"all tests passed","impossible":false}'
        )
        output = []
        async for item in self.handler._convert_anthropic_messages_stream(
            anthropic_text_stream([merged[:22], merged[22:]]),
            request_context="test-lost-prefix-stop-hook",
            request_data=stop_hook_request(),
        ):
            output.append(item.decode() if isinstance(item, bytes) else str(item))

        decision = json.loads(emitted_text("".join(output)))
        self.assertTrue(decision["ok"])
        self.assertEqual(decision["reason"], "all tests passed")
        self.assertFalse(decision["impossible"])

    async def test_stop_hook_repairs_ok_key_lost_after_thinking_delta(self):
        output = []
        async for item in self.handler._convert_anthropic_messages_stream(
            anthropic_thinking_then_text_stream(
                ["The evidence is sufficient, so the decision is ok true."],
                ['true,"reason":"all tests passed","impossible":false}'],
            ),
            request_context="test-lost-ok-key-stop-hook",
            request_data=stop_hook_request(),
        ):
            output.append(item.decode() if isinstance(item, bytes) else str(item))

        rendered = "".join(output)
        decision = json.loads(emitted_text(rendered))
        self.assertTrue(decision["ok"])
        self.assertEqual(decision["reason"], "all tests passed")
        self.assertFalse(decision["impossible"])
        self.assertEqual(rendered.count("event: message_stop\n"), 1)

    async def test_stop_hook_does_not_treat_ok_prose_as_completion(self):
        output = []
        with patch(
            "opencode_compat_hook.hook._stop_hook_json_fallback_available",
            return_value=True,
        ):
            async for item in self.handler._convert_anthropic_messages_stream(
                anthropic_text_stream(["The ok result may be true, but no JSON was returned."]),
                request_context="test-ok-prose-stop-hook",
                request_data=stop_hook_request(),
            ):
                output.append(item.decode() if isinstance(item, bytes) else str(item))

        decision = json.loads(emitted_text("".join(output)))
        self.assertFalse(decision["ok"])
        self.assertIn("not proven satisfied", decision["reason"])

    async def test_stop_hook_replaces_invalid_narrative_before_timeout(self):
        narrative = "The assistant still needs to deploy and test the firmware."
        output = []
        with (
            patch(
                "opencode_compat_hook.hook._stop_hook_json_fallback_available",
                return_value=True,
            ),
            patch(
                "opencode_compat_hook.hook._record_stop_hook_json_fallback"
            ) as record_fallback,
            patch("opencode_compat_hook.hook._stop_hook_json_fallback_due", return_value=True),
        ):
            async for item in self.handler._convert_anthropic_messages_stream(
                anthropic_text_stream([narrative]),
                request_context="test-invalid-stop-hook",
                request_data=stop_hook_request(),
            ):
                output.append(item.decode() if isinstance(item, bytes) else str(item))

        rendered = "".join(output)
        decision = json.loads(emitted_text(rendered))
        self.assertNotIn(narrative, emitted_text(rendered))
        self.assertFalse(decision["ok"])
        self.assertIn("stopping condition is not proven satisfied", decision["reason"])
        self.assertIn('"stop_reason": "end_turn"', rendered)
        record_fallback.assert_called_once()

    async def test_unclosed_think_with_content_ends_well_formed_at_message_delta(self):
        output = []
        async for item in self.handler._convert_anthropic_messages_stream(
            anthropic_text_stream(["<think>internal reasoning never closed"]),
            request_context="test-unclosed-think-end",
            request_data={"call_type": "anthropic_messages", "stream": True},
        ):
            output.append(item.decode() if isinstance(item, bytes) else str(item))

        rendered = "".join(output)
        self.assertEqual(rendered.count("event: message_stop\n"), 1)
        self.assertIn("internal reasoning never closed", emitted_text(rendered))
        self.assertNotIn("<think>", rendered)

    async def test_unclosed_think_with_content_ends_well_formed_at_finish_reason(self):
        output = []
        async for item in self.handler._convert_anthropic_messages_stream(
            openai_text_stream(["<think>internal reasoning never closed"]),
            request_context="test-unclosed-think-finish",
            request_data={"call_type": "anthropic_messages", "stream": True},
        ):
            output.append(item.decode() if isinstance(item, bytes) else str(item))

        rendered = "".join(output)
        self.assertEqual(rendered.count("event: message_stop\n"), 1)
        self.assertIn("internal reasoning never closed", emitted_text(rendered))
        self.assertNotIn("<think>", rendered)

    async def test_empty_unclosed_think_emits_placeholder_at_message_delta(self):
        output = []
        async for item in self.handler._convert_anthropic_messages_stream(
            anthropic_text_stream(["<think>"]),
            request_context="test-empty-unclosed-think",
            request_data={"call_type": "anthropic_messages", "stream": True},
        ):
            output.append(item.decode() if isinstance(item, bytes) else str(item))

        rendered = "".join(output)
        self.assertEqual(rendered.count("event: message_stop\n"), 1)
        self.assertEqual(emitted_text(rendered), ".")
        self.assertNotIn("<think>", rendered)

    async def test_empty_unclosed_think_emits_placeholder_at_finish_reason(self):
        output = []
        async for item in self.handler._convert_anthropic_messages_stream(
            openai_text_stream(["<think>"]),
            request_context="test-empty-unclosed-think-finish",
            request_data={"call_type": "anthropic_messages", "stream": True},
        ):
            output.append(item.decode() if isinstance(item, bytes) else str(item))

        rendered = "".join(output)
        self.assertEqual(rendered.count("event: message_stop\n"), 1)
        self.assertEqual(emitted_text(rendered), ".")
        self.assertNotIn("<think>", rendered)

    async def test_empty_unclosed_think_placeholder_with_closed_text_block(self):
        output = []
        async for item in self.handler._convert_anthropic_messages_stream(
            anthropic_text_with_block_stop_stream(["<think>"]),
            request_context="test-empty-unclosed-think-closed-block",
            request_data={"call_type": "anthropic_messages", "stream": True},
        ):
            output.append(item.decode() if isinstance(item, bytes) else str(item))

        rendered = "".join(output)
        self.assertEqual(rendered.count("event: message_stop\n"), 1)
        text = emitted_text(rendered)
        self.assertEqual(text, ".")
        block_starts = rendered.count("event: content_block_start\n")
        block_stops = rendered.count("event: content_block_stop\n")
        self.assertEqual(block_starts, 2)
        self.assertEqual(block_stops, 2)
        text_delta_lines = [
            line
            for line in rendered.splitlines()
            if line.startswith("data: ")
            and (json.loads(line[6:]).get("delta") or {}).get("type") == "text_delta"
        ]
        for line in text_delta_lines:
            index = json.loads(line[6:]).get("index")
            self.assertEqual(index, 1)

    async def test_unclosed_think_message_stop_without_delta_ends_well_formed(self):
        output = []
        async for item in self.handler._convert_anthropic_messages_stream(
            anthropic_text_stream_stop_only(["<think>internal reasoning never closed"]),
            request_context="test-unclosed-think-stop-only",
            request_data={"call_type": "anthropic_messages", "stream": True},
        ):
            output.append(item.decode() if isinstance(item, bytes) else str(item))

        rendered = "".join(output)
        self.assertEqual(rendered.count("event: message_stop\n"), 1)
        self.assertIn("internal reasoning never closed", emitted_text(rendered))
        self.assertNotIn("<think>", rendered)
        self.assertEqual(rendered.count("event: content_block_stop\n"), 1)
        delta_index = rendered.find("event: message_delta\n")
        stop_index = rendered.find("event: message_stop\n")
        self.assertNotEqual(delta_index, -1)
        self.assertLess(delta_index, stop_index)
        self.assertIn('"stop_reason": "end_turn"', rendered)

    async def test_empty_unclosed_think_message_stop_without_delta_emits_placeholder(self):
        output = []
        async for item in self.handler._convert_anthropic_messages_stream(
            anthropic_text_stream_stop_only(["<think>"]),
            request_context="test-empty-unclosed-think-stop-only",
            request_data={"call_type": "anthropic_messages", "stream": True},
        ):
            output.append(item.decode() if isinstance(item, bytes) else str(item))

        rendered = "".join(output)
        self.assertEqual(rendered.count("event: message_stop\n"), 1)
        self.assertEqual(emitted_text(rendered), ".")
        self.assertNotIn("<think>", rendered)
        self.assertEqual(rendered.count("event: content_block_stop\n"), 1)
        delta_index = rendered.find("event: message_delta\n")
        stop_index = rendered.find("event: message_stop\n")
        self.assertNotEqual(delta_index, -1)
        self.assertLess(delta_index, stop_index)
        self.assertIn('"stop_reason": "end_turn"', rendered)

    async def test_revealed_thinking_ends_well_formed_without_placeholder(self):
        output = []
        with patch(
            "opencode_compat_hook.hook.time.time",
            side_effect=[1_700_000_000.0] + [1_700_000_040.0] * 100,
        ):
            async for item in self.handler._convert_anthropic_messages_stream(
                anthropic_text_stream(["<think>long revealed reasoning"]),
                request_context="test-revealed-end",
                request_data={"call_type": "anthropic_messages", "stream": True},
            ):
                output.append(item.decode() if isinstance(item, bytes) else str(item))

        rendered = "".join(output)
        self.assertEqual(rendered.count("event: message_stop\n"), 1)
        self.assertIn('"stop_reason":"end_turn"', rendered)
        text = emitted_text(rendered)
        self.assertIn("long revealed reasoning", text)
        self.assertNotIn("<think>", rendered)

    async def test_unclosed_think_with_content_ends_well_formed_at_done(self):
        output = []
        async for item in self.handler._convert_anthropic_messages_stream(
            openai_stream_no_finish_reason(["<think>internal reasoning never closed"]),
            request_context="test-unclosed-think-done",
            request_data={"call_type": "anthropic_messages", "stream": True},
        ):
            output.append(item.decode() if isinstance(item, bytes) else str(item))

        rendered = "".join(output)
        self.assertEqual(rendered.count("event: message_stop\n"), 1)
        self.assertIn('"stop_reason": "end_turn"', rendered)
        self.assertIn("internal reasoning never closed", emitted_text(rendered))
        self.assertNotIn("<think>", rendered)

    async def test_empty_unclosed_think_emits_placeholder_at_done(self):
        output = []
        async for item in self.handler._convert_anthropic_messages_stream(
            openai_stream_no_finish_reason(["<think>"]),
            request_context="test-empty-unclosed-think-done",
            request_data={"call_type": "anthropic_messages", "stream": True},
        ):
            output.append(item.decode() if isinstance(item, bytes) else str(item))

        rendered = "".join(output)
        self.assertEqual(rendered.count("event: message_stop\n"), 1)
        self.assertEqual(emitted_text(rendered), ".")
        self.assertNotIn("<think>", rendered)

    async def test_unclosed_think_with_content_ends_well_formed_at_eof(self):
        output = []
        async for item in self.handler._convert_anthropic_messages_stream(
            openai_stream_no_finish_reason(
                ["<think>internal reasoning never closed"], include_done=False
            ),
            request_context="test-unclosed-think-eof",
            request_data={"call_type": "anthropic_messages", "stream": True},
        ):
            output.append(item.decode() if isinstance(item, bytes) else str(item))

        rendered = "".join(output)
        self.assertEqual(rendered.count("event: message_stop\n"), 1)
        self.assertIn('"stop_reason": "end_turn"', rendered)
        self.assertIn("internal reasoning never closed", emitted_text(rendered))
        self.assertNotIn("<think>", rendered)

    async def test_empty_unclosed_think_emits_placeholder_at_eof(self):
        output = []
        async for item in self.handler._convert_anthropic_messages_stream(
            openai_stream_no_finish_reason(["<think>"], include_done=False),
            request_context="test-empty-unclosed-think-eof",
            request_data={"call_type": "anthropic_messages", "stream": True},
        ):
            output.append(item.decode() if isinstance(item, bytes) else str(item))

        rendered = "".join(output)
        self.assertEqual(rendered.count("event: message_stop\n"), 1)
        self.assertEqual(emitted_text(rendered), ".")
        self.assertNotIn("<think>", rendered)


if __name__ == "__main__":
    unittest.main()
