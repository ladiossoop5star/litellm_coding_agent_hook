import time
import unittest

from opencode_compat_hook.hook import (
    OpencodeCompatHandler,
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


if __name__ == "__main__":
    unittest.main()
