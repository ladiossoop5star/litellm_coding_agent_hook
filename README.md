# LiteLLM OpenCode Compat Hook

A LiteLLM `CustomLogger` callback that bridges the gap between **coding agents** (OpenCode, Claude Code, Codex CLI, etc.) and **open-weight models** (DeepSeek, Qwen, GLM, etc.) that emit tool calls in non-standard formats.

## Why This Exists

Coding agents rely on structured tool calls to interact with files, shell, and other tools. Open-weight models often:

- Emit tool calls in **raw DSML** or **Qwen XML** format instead of the standard OpenAI/Anthropic API format
- **Leak internal system prompts** (e.g., `<system-reminder>`, compression artifacts) into responses
- Generate **malformed or incomplete tool calls** that crash agent parsers
- Use `<think>` blocks that need to be suppressed or revealed with appropriate timing
- Require **format bridging** between API protocols (e.g., OpenAI Responses API → chat completions, Anthropic Messages → OpenAI stream)

These issues are not just cosmetic — they cause real failures. A malformed tool call or a leaked system prompt can crash the agent's parser, bringing the entire workflow to a halt. In an unattended autonomous loop, the agent is expected to keep working until the task is complete. But these errors stop it mid-task, forcing it to wait for human intervention or fail silently. This defeats the purpose of autonomous coding agents.

This hook intercepts requests and responses at the LiteLLM proxy layer, normalizes all these edge cases, and presents clean, standards-compliant output to the coding agent — keeping the loop running.

## Quick Start

### 1. Place the package

Copy `opencode_compat_hook/` into your LiteLLM project:

```
your-litellm-project/
├── config.yaml
├── docker-compose.yml
└── opencode_compat_hook/
    ├── __init__.py
    ├── hook.py
    └── parser.py
```

### 2. Register the callback

In `config.yaml`:

```yaml
litellm_settings:
  callbacks:
    - opencode_compat_hook.hook.proxy_handler_instance
```

### 3. Mount as a volume (Docker)

In `docker-compose.yml`:

```yaml
services:
  litellm:
    image: ghcr.io/berriai/litellm:main-latest
    volumes:
      - ./config.yaml:/app/config.yaml:ro
      - ./opencode_compat_hook:/app/opencode_compat_hook:ro
    command:
      - "--config"
      - "/app/config.yaml"
      - "--port"
      - "4000"
```

### 4. Configure your coding agent

Point the agent to the LiteLLM proxy. The hook handles everything transparently.

## Tested With

This hook, together with the provided `config.yaml`, has been tested with the following coding agents and models:

| Agent | Model |
|---|---|
| OpenCode | Qwen 3.6 27B, DeepSeek V4 Flash DSpark, GLM-5.2 |
| Codex CLI | Qwen 3.6 27B, DeepSeek V4 Flash DSpark, GLM-5.2 |
| Claude Code | Qwen 3.6 27B, DeepSeek V4 Flash DSpark, GLM-5.2 |

## Features

### Raw Tool-Call Format Conversion

Converts non-standard tool-call formats into standard OpenAI `tool_calls` or Anthropic `tool_use` structures:

| Input Format | Example | Model |
|---|---|---|
| DSML | `<｜DSML｜tool_calls><invoke name="fn">...</invoke></｜DSML｜tool_calls>` | DeepSeek |
| DSML (alt bar) | `<\|DSML\|tool_calls>...</\|DSML\|tool_calls>` | DeepSeek |
| Plain XML | `<tool_calls><invoke name="fn">...</invoke></tool_calls>` | DeepSeek / Qwen |
| Qwen XML | `<tool_call><name>fn</name><parameters>{}</parameters></tool_call>` | GLM-5.2, Qwen |

Works in both **streaming** and **non-streaming** modes.

**Non-streaming support**: For non-streaming responses (`async_post_call_success_hook`), the hook scans the response content for complete raw tool blocks and converts them to standard `tool_calls`, setting `finish_reason` to `"tool_calls"`. Anthropic-style responses (dict with `content` list) are handled separately via `_convert_anthropic_message_response()`, which extracts raw tool blocks from `text` or `thinking` blocks and inserts `tool_use` entries into the content array.

### GLM-5.2 Streaming Tool Call Fixes

GLM-5.2 exhibits two specific streaming behavior issues that break coding agents:

- **`function.name=None` on continuation chunks** (`7d5e775`): GLM-5.2 sends each tool call as a sequence of stream deltas. The initial chunk carries the `function.name` and `id`, but subsequent argument-only chunks omit `function.name`, setting it to `None`. OpenCode rejects these with `Expected function.name to be a string`. The hook patches `function.name` on continuation chunks to reuse the name from the first chunk that contained it.

- **Missing `id` on continuation chunks** (`5d47192`): GLM-5.2 also omits the tool call `id` on argument-only continuation chunks. The hook tracks each tool call by its stream index (`tool_call_state: Dict[Any, Dict[str, str]]`), persists the original `call_xxx` ID and function name from the first delta, and restores them on subsequent deltas — producing a complete, consistent tool call across the entire stream.

### Hidden Thinking Management

- Suppresses `<think>...</think>` blocks from stream output
- Automatically reveals hidden thinking after 30 seconds if the block never closes
- Logs suppression statistics (chars, chunks, preview up to 200 chars)
- If a `<think>` block closes cleanly and the response has visible content, it is left undisturbed
- Raises a `RuntimeError` if an unclosed `<think>` block results in an entirely empty assistant turn (no text, no tool calls)
- Reveals thinking content that contains tool-call prefixes as hidden thinking fallback

### First-Native-Tool Stop

For DeepSeek models and Anthropic Messages streams, automatically stops the stream as soon as the first complete native tool call is detected. Prevents the model from generating additional, often spurious, tool calls after the intended one.

### Anthropic Messages API Stream Conversion

When a coding agent sends requests through the `/v1/messages` endpoint, the underlying model may only support the OpenAI chat completion format. This hook bridges the gap by converting OpenAI-style delta chunks into Anthropic SSE events:

```
message_start → content_block_start → content_block_delta
→ content_block_stop → message_delta → message_stop
```

- Supports both `text_delta` and `thinking_delta` content types
- Synthesizes complete `tool_use` content blocks (with id, name, and input) from parsed raw tool calls
- Handles OpenAI SSE mode (`data: {...}`) transparently within the Messages stream
- Estimated input tokens are injected into the `message_start` event when the upstream model does not provide them

### Stop Hook JSON Evaluator Fallback

When a request contains a Stop hook evaluator (matching `hook_event_name: Stop` with the expected JSON schema):

- Monitors the stream for visible text output
- If only reasoning is produced for more than 28 seconds, synthesizes a safe fallback:
  `{"ok": false, "reason": "No usable Stop hook JSON...", "impossible": false}`
- Tracks consecutive fallback counts per session (file-locked JSON at `/tmp/opencode_compat_stop_hook_fallback_counts.json`)
- Suppresses fallback after 5 consecutive failures; resets after 30 minutes idle

### Responses API Bridge

- **Empty tools patching**: Removes empty `tools` arrays from Responses API → chat completion conversions
- **Reasoning text preservation**: Maps `response.reasoning_text.delta` events to `reasoning_content` in the OpenAI stream
- **Disable reasoning merge**: Sets `merge_reasoning_content_in_choices = False` to prevent litellm from folding reasoning into content, which would interfere with raw tool call detection
- **Input history sanitization**: Strips raw `<think>` tags from assistant history in Responses API input
- **Malformed function call cleanup**: Removes `function_call` entries with non-JSON `arguments` and their corresponding `function_call_output` entries
- **Stream passthrough**: Skips DSML conversion for `responses`/`aresponses`/`pass_through_endpoint` call types, letting native stream chunks pass through unchanged
- **Compaction detection**: Disables tools for Codex compaction requests (identified by `x-codex-turn-metadata` or `CONTEXT CHECKPOINT COMPACTION` marker)

### Request Pre-Processing

- **Tool format normalization**: Converts Responses API tool format (`name`/`description`/`parameters`) to chat completion function-calling format
- **Tool type filtering**: Strips non-function tools, drops empty tool arrays
- **Assistant message padding**: Inserts placeholder content (`"."`) in empty assistant messages, preventing model errors. For structured content arrays, adds `{"type": "text", "text": "."}` when the array is empty or has no text/tool_use blocks
- **Internal artifact stripping**: Removes `<system-reminder>`, DSML-compression, and other internal artifacts (including `｜DSML｜` and `\|DSML\|` variants) from message history
- **Thinking-to-reasoning mapping**: Translates Anthropic `thinking.budget_tokens` to `reasoning_effort` (high/medium/low)
- **Forced streaming**: Automatically enables streaming for Anthropic Messages calls

### Additional API Routes

| Route | Purpose |
|---|---|
| `POST /v1/responses/input_tokens` | Estimates input tokens by character count |
| `POST /v1/messages/count_tokens` | Delegates to native endpoint, falls back to local estimation (CJK-aware, includes message/tool overhead) |

### Built-in Token Estimator

Used when the upstream model does not provide token counts. The estimator accounts for:
- **CJK characters** (1 token each) vs. **other characters** (1 token per 4 characters)
- **Message framing overhead**: 128 base + 12 per message + 24 per tool
- **Safety margin**: 10% buffer on all estimates

### Stream Keepalive

- Sends keepalive signals after 15 seconds of silence (5 seconds for Stop hook evaluator)
- Automatically terminates after 600 seconds of idle time

### LiteLLM Monkey-Patches (Runtime Only)

These patches are applied at runtime and do not modify any source files:

- `_patch_litellm_responses_empty_tools_bridge` — Fixes Responses → chat conversion empty tools
- `_patch_litellm_responses_reasoning_text_bridge` — Preserves reasoning text deltas in Responses stream
- `_patch_litellm_client_disconnect_metadata` — Handles None metadata in disconnect tracking
