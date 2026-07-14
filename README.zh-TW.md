# LiteLLM OpenCode Compat Hook

一個 LiteLLM `CustomLogger` callback，橋接 **coding agent**（OpenCode、Claude Code、Codex CLI 等）與**開放權重模型**（DeepSeek、Qwen、GLM 等）之間的 tool-call 相容性問題。

## 為什麼需要這個

Coding agent 依賴結構化的工具呼叫來操作檔案、Shell 和其他工具。開放權重模型經常：

- 使用**原始 DSML 或 Qwen XML 格式**輸出工具呼叫，而非標準的 OpenAI/Anthropic API 格式
- 將**內部系統提示**（如 `<system-reminder>`、壓縮 artifact）洩漏到回應中
- 產生**格式錯誤或不完整**的工具呼叫，導致 agent 解析器崩潰
- 使用 `<think>` 區塊，需要以適當的時機壓制或揭露
- 需要在不同 API 協定之間進行**格式橋接**（例如 OpenAI Responses API ↔ chat completions、Anthropic Messages ↔ OpenAI stream）

這些問題不只是表面上的格式異常，而是會造成實際的運作中斷。格式錯誤的工具呼叫或外洩的系統提示，會直接讓 agent 的解析器崩潰，導致整個工作流程停擺。在無人監守的自動化 loop 中，agent 本該持續工作直到任務完成。但這些錯誤會讓它中途停下來，等待人工介入或默默地失敗——這完全違背了自主 coding agent 的設計目的。

這個 hook 在 LiteLLM proxy 層攔截請求和回應，將所有這些邊界情況正規化，向 coding agent 呈現乾淨、符合標準的輸出——讓 loop 持續運轉。

## 快速開始

### 1. 放置套件

將 `opencode_compat_hook/` 複製到你的 LiteLLM 專案中：

```
your-litellm-project/
├── config.yaml
├── docker-compose.yml
└── opencode_compat_hook/
    ├── __init__.py
    ├── hook.py
    └── parser.py
```

### 2. 註冊 callback

在 `config.yaml` 中：

```yaml
litellm_settings:
  callbacks:
    - opencode_compat_hook.hook.proxy_handler_instance
```

### 3. 掛載 volume（Docker）

在 `docker-compose.yml` 中：

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

### 4. 設定 coding agent

將 agent 指向 LiteLLM proxy，hook 會透明處理所有問題。

## 測試過的組合

此 hook 搭配提供的 `config.yaml`，已在以下 coding agent 與模型組合上測試通過：

| Agent | 模型 |
|---|---|
| OpenCode | Qwen 3.6 27B、DeepSeek V4 Flash DSpark、GLM-5.2 |
| Codex CLI | Qwen 3.6 27B、DeepSeek V4 Flash DSpark、GLM-5.2 |
| Claude Code | Qwen 3.6 27B、DeepSeek V4 Flash DSpark、GLM-5.2 |

## 功能列表

### 原始工具呼叫格式轉換

將非標準的工具呼叫格式轉換為標準的 OpenAI `tool_calls` 或 Anthropic `tool_use` 結構：

| 輸入格式 | 範例 | 適用模型 |
|---|---|---|
| DSML | `<｜DSML｜tool_calls><invoke name="fn">...</invoke></｜DSML｜tool_calls>` | DeepSeek |
| DSML（替代分隔符） | `<\|DSML\|tool_calls>...</\|DSML\|tool_calls>` | DeepSeek |
| 簡化 XML | `<tool_calls><invoke name="fn">...</invoke></tool_calls>` | DeepSeek / Qwen |
| Qwen XML | `<tool_call><name>fn</name><parameters>{}</parameters></tool_call>` | GLM-5.2, Qwen |

支援**串流**與**非串流**兩種模式。

**非串流支援**：對非串流回應（`async_post_call_success_hook`），hook 掃描回應內容中完整的原始工具區塊，轉換為標準 `tool_calls`，並將 `finish_reason` 設為 `"tool_calls"`。Anthropic-style 回應（含 `content` list 的 dict）則透過 `_convert_anthropic_message_response()` 處理，從 `text` 或 `thinking` 區塊中擷取原始工具區塊，在 content 陣列中插入 `tool_use` 項目。

### GLM-5.2 串流工具呼叫修復

GLM-5.2 有兩個特定的串流行為會導致 coding agent 異常：

- **`function.name=None` 在續傳 chunk 中**（`7d5e775`）：GLM-5.2 將每個工具呼叫拆分為多個串流 delta。第一個 chunk 帶有 `function.name` 和 `id`，但後續只傳 arguments 的 chunk 會省略 `function.name`，設為 `None`。OpenCode 會因此拋出 `Expected function.name to be a string`。Hook 將續傳 chunk 的 `function.name` 補回第一個 chunk 中的值。

- **續傳 chunk 缺少 `id`**（`5d47192`）：GLM-5.2 在只傳 arguments 的續傳 chunk 中也會省略 tool call 的 `id`。Hook 透過 stream index 追蹤每個工具呼叫（`tool_call_state: Dict[Any, Dict[str, str]]`），從第一個 delta 保存原始的 `call_xxx` ID 和 function name，並在後續 delta 中還原 — 產出完整一致的 tool call。

### 隱藏思考過程管理

- 壓制 `<think>...</think>` 區塊不在串流中輸出
- 30 秒後若思考區塊仍未結束，自動揭露隱藏內容
- 記錄壓制統計資料（字元數、chunk 數、前 200 字預覽）
- 若 `<think>` 正常關閉且回應有可見內容，則不干擾
- 若未關閉的 `<think>` 導致完全空的 assistant turn（無文字、無工具呼叫），拋出 `RuntimeError`
- 若隱藏思考中出現工具呼叫前綴，自動揭露做為回退

### 第一個 Native Tool 後停止

對 DeepSeek 模型和 Anthropic Messages 串流，偵測到第一個完整的原生 tool call 後立即停止串流，避免模型產生多餘的工具呼叫。

### Anthropic Messages API 串流轉換

當 coding agent 透過 `/v1/messages` 端點發送請求時，底層模型可能只支援 OpenAI chat completion 格式。此 hook 橋接 gap，將 OpenAI-style delta chunk 轉換為 Anthropic SSE 事件順序：

```
message_start → content_block_start → content_block_delta
→ content_block_stop → message_delta → message_stop
```

- 支援 `text_delta` 與 `thinking_delta` 兩種內容類型
- 從解析出的原始工具呼叫合成完整的 `tool_use` content block（含 id、name、input）
- 在 Messages 串流中透明處理 OpenAI SSE 模式（`data: {...}`）
- 當上游模型未提供 input tokens 時，在 `message_start` 事件中注入估算值

### Stop Hook JSON Evaluator 回退

當請求包含 Stop hook evaluator（匹配 `hook_event_name: Stop` 及預期的 JSON schema）：

- 監控串流是否有可見文字輸出
- 若超過 28 秒只有推理內容，自動合成安全回退：
  `{"ok": false, "reason": "No usable Stop hook JSON...", "impossible": false}`
- 每個 session 記錄連續回退次數（檔案鎖 JSON 位於 `/tmp/opencode_compat_stop_hook_fallback_counts.json`）
- 連續 5 次失敗後暫停回退；30 分鐘閒置後重置

### Responses API 橋接

- **空工具修補**：移除 Responses API → chat completion 轉換中的空 `tools` 陣列
- **推理文字保留**：將 `response.reasoning_text.delta` 事件映射為 OpenAI 串流中的 `reasoning_content`
- **停用 reasoning 合併**：設 `merge_reasoning_content_in_choices = False`，防止 litellm 把 reasoning 合進 content 干擾原始工具呼叫偵測
- **輸入歷史清理**：從 Responses API 輸入中剝離原始的 `<think>` 標籤
- **格式異常 function call 清理**：移除 arguments 非 JSON 的 `function_call` 及其對應的 `function_call_output`
- **串流跳過**：對 `responses`/`aresponses`/`pass_through_endpoint` 類型跳過 DSML 轉換，讓原生串流 chunk 直接通過
- **Compaction 偵測**：對 Codex compaction 請求自動關閉工具（透過 `x-codex-turn-metadata` 或 `CONTEXT CHECKPOINT COMPACTION` 標記辨識）

### 請求前處理

- **工具格式正規化**：將 Responses API 工具格式（`name`/`description`/`parameters`）轉為 chat completion function-calling 格式
- **工具類型過濾**：移除非 function 類型的工具，刪除空陣列
- **Assistant 訊息填充**：在空的 assistant message 中插入佔位符內容 `"."`，防止模型錯誤。對結構化 content 陣列，當陣列為空或無 text/tool_use 區塊時，補入 `{"type": "text", "text": "."}`
- **內部 artifact 剝離**：從訊息歷史中移除 `<system-reminder>`、DSML 壓縮等內部 artifact（包括 `｜DSML｜` 與 `\|DSML\|` 變體）
- **Thinking 轉 reasoning 映射**：將 Anthropic `thinking.budget_tokens` 轉譯為 `reasoning_effort`（高/中/低）
- **強制串流**：對 Anthropic Messages 呼叫自動啟用串流

### 額外 API 路由

| 路由 | 用途 |
|---|---|
| `POST /v1/responses/input_tokens` | 以字元數估算 input tokens |
| `POST /v1/messages/count_tokens` | 優先呼叫原生 endpoint，失敗時使用本地估算（支援 CJK、含 message/tool overhead） |

### 串流 Keepalive

- 每隔 15 秒無資料發送 keepalive（Stop hook evaluator 模式為 5 秒）
- 閒置 600 秒後自動中斷

### 內建 Token 估算器

當上游模型未提供 token 計數時使用。估算考量：
- **CJK 字元**（每字 1 token）vs **其他字元**（每 4 字 1 token）
- **訊息框架開銷**：128 基礎 + 每則訊息 12 + 每個工具 24
- **安全邊際**：所有估算加 10% 緩衝

### LiteLLM 執行時期修補（僅記憶體）

這些修補在執行時期動態套用，不修改任何原始碼：

- `_patch_litellm_responses_empty_tools_bridge` — 修復 Responses → chat 轉換的空工具問題
- `_patch_litellm_responses_reasoning_text_bridge` — 保留 Responses 串流中的推理文字 delta
- `_patch_litellm_client_disconnect_metadata` — 處理斷線追蹤中的 None metadata
