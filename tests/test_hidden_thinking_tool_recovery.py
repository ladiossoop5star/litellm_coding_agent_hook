import json
import time
import unittest
from unittest.mock import patch

from opencode_compat_hook.hook import (
    OpencodeCompatHandler,
    _is_stop_hook_json_evaluator,
    _raw_think_state,
    _request_tool_schemas,
)


BASH_SCHEMA = {
    "type": "object",
    "properties": {
        "command": {"type": "string"},
        "description": {"type": "string"},
    },
    "required": ["command", "description"],
    "additionalProperties": False,
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


class HiddenThinkingToolRecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.handler = object.__new__(OpencodeCompatHandler)

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


if __name__ == "__main__":
    unittest.main()
