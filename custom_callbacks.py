import litellm
from litellm.integrations.custom_logger import CustomLogger
from typing import AsyncGenerator

def update_usage_dict(usage_dict, input_tokens):
    if not isinstance(usage_dict, dict):
        return
    # Map input_tokens
    if usage_dict.get("input_tokens") is None or usage_dict.get("input_tokens") == 0:
        usage_dict["input_tokens"] = input_tokens
    if usage_dict.get("prompt_tokens") is None or usage_dict.get("prompt_tokens") == 0:
        usage_dict["prompt_tokens"] = input_tokens
    
    # Calculate total_tokens
    out_tokens = usage_dict.get("output_tokens") or usage_dict.get("completion_tokens") or 0
    usage_dict["total_tokens"] = usage_dict["input_tokens"] + out_tokens

def update_usage_obj(usage_obj, input_tokens):
    # Check if has input_tokens attribute
    if hasattr(usage_obj, "input_tokens"):
        if getattr(usage_obj, "input_tokens") is None or getattr(usage_obj, "input_tokens") == 0:
            setattr(usage_obj, "input_tokens", input_tokens)
    if hasattr(usage_obj, "prompt_tokens"):
        if getattr(usage_obj, "prompt_tokens") is None or getattr(usage_obj, "prompt_tokens") == 0:
            setattr(usage_obj, "prompt_tokens", input_tokens)
            
    out_tokens = getattr(usage_obj, "output_tokens", None) or getattr(usage_obj, "completion_tokens", None) or 0
    if hasattr(usage_obj, "total_tokens"):
        in_tokens = getattr(usage_obj, "input_tokens", None) or getattr(usage_obj, "prompt_tokens", None) or 0
        setattr(usage_obj, "total_tokens", int(in_tokens) + int(out_tokens))

def clean_tools(tools):
    if not isinstance(tools, list):
        return tools
    
    cleaned = []
    for tool in tools:
        if not isinstance(tool, dict):
            cleaned.append(tool)
            continue
        
        t_type = tool.get("type")
        if t_type == "function":
            if "function" in tool:
                cleaned.append(tool)
            elif "name" in tool:
                wrapped = {
                    "type": "function",
                    "function": {
                        "name": tool.get("name"),
                        "description": tool.get("description"),
                        "parameters": tool.get("parameters"),
                    }
                }
                if "strict" in tool:
                    wrapped["function"]["strict"] = tool["strict"]
                cleaned.append(wrapped)
            else:
                cleaned.append(tool)
        elif t_type == "namespace":
            nested_tools = tool.get("tools", [])
            if isinstance(nested_tools, list):
                for nt in nested_tools:
                    if not isinstance(nt, dict):
                        continue
                    name = nt.get("name")
                    if name:
                        wrapped = {
                            "type": "function",
                            "function": {
                                "name": name,
                                "description": nt.get("description"),
                                "parameters": nt.get("parameters"),
                            }
                        }
                        if "strict" in nt:
                            wrapped["function"]["strict"] = nt["strict"]
                        cleaned.append(wrapped)
        elif t_type == "custom":
            pass
        else:
            pass
    return cleaned

class ResponseUsageCallback(CustomLogger):
    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        """
        在請求發送前，預先計算 input tokens
        """
        try:
            if isinstance(data, dict) and "tools" in data:
                data["tools"] = clean_tools(data["tools"])
        except Exception:
            pass

        try:
            model = data.get("model", "")
            messages = data.get("messages", []) or data.get("input", []) or []
            
            # 安全計算 token 數
            try:
                input_tokens = litellm.token_counter(model=model, messages=messages)
            except Exception:
                try:
                    # 降級使用 gpt-3.5-turbo token 估算
                    input_tokens = litellm.token_counter(model="gpt-3.5-turbo", messages=messages)
                except Exception:
                    # 字元數估算 (4 字元約 1 token)
                    total_chars = 0
                    for m in messages:
                        if isinstance(m, dict):
                            content = m.get("content", "")
                            if isinstance(content, str):
                                total_chars += len(content)
                            elif isinstance(content, list):
                                for part in content:
                                    if isinstance(part, dict) and part.get("type") == "text":
                                        total_chars += len(part.get("text", ""))
                    input_tokens = max(1, total_chars // 4)
            
            data["_custom_input_tokens"] = input_tokens
        except Exception:
            pass
        return data

    async def async_post_call_success_hook(self, data, user_api_key_dict, response):
        """
        處理非串流回應：注入 input_tokens
        """
        try:
            calculated_input_tokens = data.get("_custom_input_tokens")
            if not calculated_input_tokens:
                return response
            
            if isinstance(response, dict):
                if "usage" in response and response["usage"]:
                    update_usage_dict(response["usage"], calculated_input_tokens)
            elif hasattr(response, "usage") and response.usage:
                if isinstance(response.usage, dict):
                    update_usage_dict(response.usage, calculated_input_tokens)
                else:
                    update_usage_obj(response.usage, calculated_input_tokens)
        except Exception:
            pass
        return response

    async def async_post_call_streaming_iterator_hook(
        self, user_api_key_dict, response: AsyncGenerator, request_data: dict
    ) -> AsyncGenerator:
        """
        處理串流回應：包裝 AsyncGenerator，攔截 Completed Event 注入 usage
        """
        calculated_input_tokens = request_data.get("_custom_input_tokens")
        
        async for chunk in response:
            try:
                if calculated_input_tokens:
                    # 1. 處理直接包含 usage 的 chunk
                    if hasattr(chunk, "usage") and chunk.usage is not None:
                        if isinstance(chunk.usage, dict):
                            update_usage_dict(chunk.usage, calculated_input_tokens)
                        else:
                            update_usage_obj(chunk.usage, calculated_input_tokens)
                    elif isinstance(chunk, dict) and "usage" in chunk and chunk["usage"] is not None:
                        update_usage_dict(chunk["usage"], calculated_input_tokens)
                    
                    # 2. 處理 ResponseCompletedEvent 等內嵌 response 的 chunk
                    if hasattr(chunk, "response") and chunk.response is not None:
                        resp_obj = chunk.response
                        if hasattr(resp_obj, "usage") and resp_obj.usage is not None:
                            if isinstance(resp_obj.usage, dict):
                                update_usage_dict(resp_obj.usage, calculated_input_tokens)
                            else:
                                update_usage_obj(resp_obj.usage, calculated_input_tokens)
                        elif isinstance(resp_obj, dict) and "usage" in resp_obj and resp_obj["usage"] is not None:
                            update_usage_dict(resp_obj["usage"], calculated_input_tokens)
            except Exception:
                pass
            yield chunk

# 實例化供 config.yaml 載入
response_usage_callback = ResponseUsageCallback()
