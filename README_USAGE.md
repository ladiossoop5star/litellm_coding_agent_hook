# LiteLLM Proxy 使用與設定說明文件

本文件說明如何使用此 LiteLLM Proxy 服務，包含常用 API 端點調用範例、可能需要修改的設定參數。

## 🔑 認證資訊
* **API Key**：`wpd-local-llm`
* **HTTP Header**：`Authorization: Bearer wpd-local-llm`

---

## 🚀 服務管理腳本

我們已於 `/opt/litellm/` 目錄下提供三個一鍵管理腳本：

* **啟動服務**：`./start.sh`  
* **停止服務**：`./stop.sh`  
* **重啟服務**：`./restart.sh`  

---

## ⚙️ 常見需調整的參數 (config.yaml)

若未來 upstream API 或對外模型名稱需要變更，請直接編輯 `/opt/litellm/config.yaml` 內之欄位：

### 1. 模型對外名稱與上游設定
在 `model_list` 區段下：
* `model_name`：LiteLLM **對外顯示的模型名稱**（例如：`claude-sonnet-4-6`）。
* `litellm_params.model`：**實際傳送給 upstream 的模型 ID**（例如：`openai/qwen3.6-27b`）。
* `litellm_params.api_base`：**上游 API 的端點 URL**（例如：`http://10.115.140.130:9526/v1`）。
* `litellm_params.api_key`：**上游的 API Key**，目前預設為 `dummy`。

### 2. Token 限制 (Model Info)
如果上游模型更新或更換，您需要調整以下欄位：
* `litellm_params.max_tokens`：單次輸出的 Token 限制。
* `litellm_params.model_info.max_tokens`：模型總 Token容量限制。
* `litellm_params.model_info.max_input_tokens`：單次輸入的 Token 限制。
* `litellm_params.model_info.max_output_tokens`：單次輸出的 Token 限制。

### 3. 全域代理路由設定 (litellm_settings)
在 `litellm_settings` 區段下，有一個關鍵參數：
* `use_chat_completions_url_for_anthropic_messages: true`：
  **用途**：強制將 Anthropic `/v1/messages` 請求導向常規的 `/v1/chat/completions` 分支做轉換。
  **重要性**：這能完美繞過 LiteLLM 對於自訂 OpenAI 端點（`custom_endpoint`）在 `Responses API` 下的物件類型轉譯 Bug，確保 content 欄位與 thinking 欄位能完整輸出而不遺漏。

---

## 🧪 常用 API 端點使用範例 (curl)

請於終端機執行以下命令以呼叫已封裝的服務：

### 1. GET /health (服務健康狀態)
```bash
curl -s http://127.0.0.1:4000/health \
  -H "Authorization: Bearer wpd-local-llm" | jq .
```

### 2. GET /v1/models (可用模型列表)
```bash
curl -s http://127.0.0.1:4000/v1/models \
  -H "Authorization: Bearer wpd-local-llm" | jq .
```

### 3. POST /v1/chat/completions (OpenAI 相容對話)
```bash
curl -s http://127.0.0.1:4000/v1/chat/completions \
  -H "Authorization: Bearer wpd-local-llm" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4-6",
    "max_tokens": 128,
    "messages": [{"role": "user", "content": "Say OK only."}]
  }' | jq .
```

### 4. POST /v1/messages (Anthropic 相容對話)
```bash
curl -s http://127.0.0.1:4000/v1/messages \
  -H "Authorization: Bearer wpd-local-llm" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-opus-4-8",
    "max_tokens": 128,
    "messages": [{"role": "user", "content": "Say OK only."}]
  }' | jq .
```

### 5. POST /v1/responses (Unified Response 端點)
```bash
curl -s http://127.0.0.1:4000/v1/responses \
  -H "Authorization: Bearer wpd-local-llm" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4-6",
    "input": "Say OK only.",
    "max_output_tokens": 128
  }' | jq .
```

### 6. POST /v1/messages/count_tokens (計算 Token 數)
```bash
curl -s http://127.0.0.1:4000/v1/messages/count_tokens \
  -H "Authorization: Bearer wpd-local-llm" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-opus-4-8",
    "messages": [{"role": "user", "content": "hello"}]
  }' | jq .
```
