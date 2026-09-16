import asyncio
import copy

from litellm.litellm_core_utils.prompt_templates.common_utils import (
    TOOL_RESULT_IMAGE_BOUNDARY,
    TOOL_RESULT_IMAGE_HISTORY_PLACEHOLDER,
    TOOL_RESULT_IMAGE_PLACEHOLDER,
)
from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import (
    LiteLLMAnthropicMessagesAdapter,
)
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig


DATA = "iVBORw0KGgoAAAANSUhEUg=="
DATA_URI = "data:image/png;base64," + DATA


def image_block(data=DATA):
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": data},
    }


def run_pipeline(results):
    messages = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": tool_id,
                    "name": "Read",
                    "input": {"file_path": "/opt/temp/pr.png"},
                }
                for tool_id, _ in results
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": tool_id, "content": content}
                for tool_id, content in results
            ],
        },
    ]
    translated = LiteLLMAnthropicMessagesAdapter().translate_anthropic_messages_to_openai(messages)
    request = OpenAIGPTConfig().transform_request(
        model="vision-model",
        messages=translated,
        optional_params={},
        litellm_params={},
        headers={},
    )
    return translated, request["messages"]


def tool_messages(messages):
    return [message for message in messages if message.get("role") == "tool"]


def image_urls_in_user_messages(messages):
    return [
        part["image_url"]["url"]
        for message in messages
        if message.get("role") == "user" and isinstance(message.get("content"), list)
        for part in message["content"]
        if isinstance(part, dict) and part.get("type") == "image_url"
    ]


def assert_no_base64_text(messages):
    for message in messages:
        content = message.get("content")
        assert not (isinstance(content, str) and content.startswith("data:image/"))


def openai_image_tool_turn(tool_id, data, text=None):
    content = []
    if text is not None:
        content.append({"type": "text", "text": text})
    content.append(
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64," + data},
        }
    )
    return [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": tool_id,
                    "type": "function",
                    "function": {"name": "Read", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": tool_id, "content": content},
    ]


def test_single_image_is_structured_then_hoisted():
    translated, outbound = run_pipeline([("toolu_01", [image_block()])])

    assert translated[1]["content"] == [
        {"type": "image_url", "image_url": {"url": DATA_URI}}
    ]
    assert [message["role"] for message in outbound] == ["assistant", "tool", "user"]
    assert tool_messages(outbound)[0]["content"] == TOOL_RESULT_IMAGE_PLACEHOLDER
    assert image_urls_in_user_messages(outbound) == [DATA_URI]
    assert outbound[-1]["content"][0] == {
        "type": "text",
        "text": TOOL_RESULT_IMAGE_BOUNDARY,
    }
    assert_no_base64_text(outbound)


def test_text_and_image_preserves_text_and_structured_image():
    _, outbound = run_pipeline(
        [("toolu_01", [{"type": "text", "text": "image follows"}, image_block()])]
    )

    assert tool_messages(outbound)[0]["content"] == [
        {"type": "text", "text": "image follows"}
    ]
    assert image_urls_in_user_messages(outbound) == [DATA_URI]
    assert_no_base64_text(outbound)


def test_multiple_images_survive():
    _, outbound = run_pipeline([("toolu_01", [image_block("ONE"), image_block("TWO")])])

    assert image_urls_in_user_messages(outbound) == [
        "data:image/png;base64,ONE",
        "data:image/png;base64,TWO",
    ]
    assert_no_base64_text(outbound)


def test_parallel_tool_results_keep_adjacency_and_order():
    _, outbound = run_pipeline(
        [("toolu_01", [image_block("ONE")]), ("toolu_02", [image_block("TWO")])]
    )

    assert [message["role"] for message in outbound] == ["assistant", "tool", "tool", "user"]
    assert [message["tool_call_id"] for message in tool_messages(outbound)] == [
        "toolu_01",
        "toolu_02",
    ]
    assert image_urls_in_user_messages(outbound) == [
        "data:image/png;base64,ONE",
        "data:image/png;base64,TWO",
    ]
    assert_no_base64_text(outbound)


def test_plain_text_tool_result_is_unchanged():
    translated, outbound = run_pipeline(
        [("toolu_01", [{"type": "text", "text": "plain result"}])]
    )

    assert translated == outbound
    assert tool_messages(outbound)[0]["content"] == "plain result"
    assert image_urls_in_user_messages(outbound) == []


def test_async_transform_hoists_image():
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "toolu_01",
                    "type": "function",
                    "function": {"name": "Read", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "toolu_01",
            "content": [{"type": "image_url", "image_url": {"url": DATA_URI}}],
        },
    ]

    outbound = asyncio.run(
        OpenAIGPTConfig()._transform_messages(messages, "vision-model", is_async=True)
    )

    assert [message["role"] for message in outbound] == ["assistant", "tool", "user"]
    assert tool_messages(outbound)[0]["content"] == TOOL_RESULT_IMAGE_PLACEHOLDER
    assert image_urls_in_user_messages(outbound) == [DATA_URI]
    assert_no_base64_text(outbound)


def test_only_newest_image_tool_run_is_hoisted():
    messages = [
        *openai_image_tool_turn("toolu_old", "OLD"),
        {"role": "assistant", "content": "I analyzed the old image."},
        {"role": "user", "content": "Read another image."},
        *openai_image_tool_turn("toolu_new", "NEW"),
    ]

    outbound = OpenAIGPTConfig()._transform_messages(messages, "vision-model")

    assert image_urls_in_user_messages(outbound) == ["data:image/png;base64,NEW"]
    tools = tool_messages(outbound)
    assert tools[0]["content"] == TOOL_RESULT_IMAGE_HISTORY_PLACEHOLDER
    assert tools[1]["content"] == TOOL_RESULT_IMAGE_PLACEHOLDER
    assert_no_base64_text(outbound)


def test_historical_mixed_tool_result_keeps_text_and_marks_omitted_image():
    messages = [
        *openai_image_tool_turn("toolu_old", "OLD", text="old image notes"),
        {"role": "assistant", "content": "Old image handled."},
        {"role": "user", "content": "Read another image."},
        *openai_image_tool_turn("toolu_new", "NEW"),
    ]

    outbound = OpenAIGPTConfig()._transform_messages(messages, "vision-model")

    assert tool_messages(outbound)[0]["content"] == [
        {"type": "text", "text": "old image notes"},
        {"type": "text", "text": TOOL_RESULT_IMAGE_HISTORY_PLACEHOLDER},
    ]
    assert image_urls_in_user_messages(outbound) == ["data:image/png;base64,NEW"]


def test_twelve_historical_images_collapse_to_latest_image():
    messages = []
    for index in range(12):
        messages.extend(openai_image_tool_turn(f"toolu_{index}", f"IMAGE_{index}"))
        if index < 11:
            messages.extend(
                [
                    {"role": "assistant", "content": f"Analyzed image {index}."},
                    {"role": "user", "content": "Continue."},
                ]
            )
    original_messages = copy.deepcopy(messages)

    outbound = OpenAIGPTConfig()._transform_messages(messages, "vision-model")

    assert image_urls_in_user_messages(outbound) == ["data:image/png;base64,IMAGE_11"]
    tools = tool_messages(outbound)
    assert [tool["content"] for tool in tools[:-1]] == [
        TOOL_RESULT_IMAGE_HISTORY_PLACEHOLDER
    ] * 11
    assert tools[-1]["content"] == TOOL_RESULT_IMAGE_PLACEHOLDER
    assert messages == original_messages
