import html
import json
import logging
import re
import uuid
from typing import Any, Dict, List, Optional


log = logging.getLogger("opencode_compat_hook")

DSML_BAR = chr(0xFF5C)
DSML_OPEN = "<" + DSML_BAR + "DSML" + DSML_BAR + "tool_calls>"
DSML_CLOSE = "</" + DSML_BAR + "DSML" + DSML_BAR + "tool_calls>"
DSML_OPEN_ALT = "<|DSML|tool_calls>"
DSML_CLOSE_ALT = "</|DSML|tool_calls>"
DSML_TAG_PREFIX = "<" + DSML_BAR + "DSML" + DSML_BAR
DSML_CLOSE_PREFIX = "</" + DSML_BAR + "DSML" + DSML_BAR
DSML_TAG_PREFIX_ALT = "<|DSML|"
DSML_CLOSE_PREFIX_ALT = "</|DSML|"

RAW_TOOL_OPEN_MARKERS = (
    DSML_OPEN,
    "<DSML>tool_calls>",
    "<tool_calls>",
    "<tool_calls",
    "<tool_call>",
    "<tool_call",
)

RAW_TOOL_FRAGMENT_MARKERS = RAW_TOOL_OPEN_MARKERS + (
    DSML_CLOSE,
    DSML_TAG_PREFIX + "invoke",
    DSML_TAG_PREFIX + "parameter",
    DSML_TAG_PREFIX + "system-reminder",
    DSML_CLOSE_PREFIX + "invoke",
    DSML_CLOSE_PREFIX + "parameter",
    DSML_CLOSE_PREFIX + "tool_calls",
    DSML_CLOSE_PREFIX + "system-reminder",
    "<DSML:",
    "</DSML:",
    "<|DSML|invoke",
    "<|DSML|parameter",
    "<|DSML|system-reminder",
    "</|DSML|invoke",
    "</|DSML|parameter",
    "</|DSML|tool_calls",
    "</|DSML|system-reminder",
    "<dcp-system-reminder>",
    "</dcp-system-reminder>",
    "<dcp-message-id>",
    "</dcp-message-id>",
    "</tool_calls>",
    "</tool_call>",
)

INTERNAL_LEAK_SENTINELS = (
    "Active compressed blocks in this session:",
    "If your selected compression range includes any listed block",
    "required placeholder exactly once in the summary using",
)


def _normalize_dsml_bars(text: str) -> str:
    return (
        text.replace(DSML_OPEN_ALT, DSML_OPEN)
        .replace(DSML_CLOSE_ALT, DSML_CLOSE)
        .replace(DSML_TAG_PREFIX_ALT + "invoke", DSML_TAG_PREFIX + "invoke")
        .replace(DSML_TAG_PREFIX_ALT + "parameter", DSML_TAG_PREFIX + "parameter")
        .replace(DSML_TAG_PREFIX_ALT + "system-reminder", DSML_TAG_PREFIX + "system-reminder")
        .replace(DSML_CLOSE_PREFIX_ALT + "invoke", DSML_CLOSE_PREFIX + "invoke")
        .replace(DSML_CLOSE_PREFIX_ALT + "parameter", DSML_CLOSE_PREFIX + "parameter")
        .replace(DSML_CLOSE_PREFIX_ALT + "system-reminder", DSML_CLOSE_PREFIX + "system-reminder")
    )


def _first_marker_index(text: str, markers: tuple[str, ...]) -> int:
    indexes = [idx for marker in markers if (idx := text.find(marker)) != -1]
    return min(indexes) if indexes else -1


def _orphan_fragment_start(text: str, marker_idx: int) -> int:
    """Suppress malformed DSML fragments from the start of their buffered text."""
    for sentinel in INTERNAL_LEAK_SENTINELS:
        idx = text.rfind(sentinel, 0, marker_idx)
        if idx != -1:
            return text.rfind("\n", 0, idx) + 1
    return 0


def normalize_raw_tool_calls(text: str) -> str:
    """Normalize supported DSML/Qwen XML variants to the canonical DSML form."""
    bar = DSML_BAR
    text = _normalize_dsml_bars(text)

    if "<DSML>tool_calls>" in text:
        text = text.replace("<DSML>tool_calls>", "<" + bar + "DSML" + bar + "tool_calls>", 1)
        text = re.sub(r"</DSML[:\s]+tool_calls\s*>", "</" + bar + "DSML" + bar + "tool_calls>", text)
        text = re.sub(r"<DSML[:\s]+(invoke)\s+", "<" + bar + "DSML" + bar + r"\1 ", text)
        text = re.sub(r"<DSML[:\s]+(parameter)\s+", "<" + bar + "DSML" + bar + r"\1 ", text)
        text = re.sub(r"</DSML[:\s]+(invoke|parameter)\s*>", "</" + bar + "DSML" + bar + r"\1>", text)
    elif "<tool_calls>" in text and "</tool_calls>" in text:
        text = text.replace("<tool_calls>", "<" + bar + "DSML" + bar + "tool_calls>", 1)
        text = text.replace("</tool_calls>", "</" + bar + "DSML" + bar + "tool_calls>", 1)
        text = re.sub(r"<invoke\s+", "<" + bar + "DSML" + bar + "invoke ", text)
        text = re.sub(r"</invoke\s*>", "</" + bar + "DSML" + bar + "invoke>", text)
        text = re.sub(r"<parameter\s+", "<" + bar + "DSML" + bar + "parameter ", text)
        text = re.sub(r"</parameter\s*>", "</" + bar + "DSML" + bar + "parameter>", text)
    return text


def normalize_arg_value(val: Any) -> str:
    if isinstance(val, str):
        val = html.unescape(val)
        try:
            parsed = json.loads(val)
            if isinstance(parsed, (dict, list)):
                return json.dumps(parsed, ensure_ascii=False)
        except (json.JSONDecodeError, ValueError):
            pass
        return val
    return json.dumps(val, ensure_ascii=False)


def make_tool_call(name: str, arguments: Any, call_id: Optional[str] = None) -> Dict[str, Any]:
    return {
        "id": call_id or "call_" + uuid.uuid4().hex[:24],
        "type": "function",
        "function": {
            "name": name,
            "arguments": normalize_arg_value(arguments),
        },
    }


def has_complete_raw_tool_block(text: str) -> bool:
    text = _normalize_dsml_bars(text)
    if DSML_OPEN in text and DSML_CLOSE in text:
        return True
    if "<DSML>tool_calls>" in text:
        return True
    if "<tool_calls>" in text and "</tool_calls>" in text:
        return True
    if "<tool_call>" in text and "</tool_call>" in text:
        return True
    return False


def has_any_dsml_prefix(text: str) -> bool:
    if not text:
        return False
    text = _normalize_dsml_bars(text)
    for marker in RAW_TOOL_FRAGMENT_MARKERS:
        if marker in text:
            return True

    tail = text[-150:] if len(text) > 150 else text
    for marker in RAW_TOOL_FRAGMENT_MARKERS:
        max_size = min(len(tail), len(marker) - 1)
        for size in range(max_size, 2, -1):
            if marker.startswith(tail[-size:]):
                return True
    return False


def find_raw_tool_start(text: str) -> int:
    text = _normalize_dsml_bars(text)
    open_idx = _first_marker_index(text, RAW_TOOL_OPEN_MARKERS)
    fragment_idx = _first_marker_index(text, RAW_TOOL_FRAGMENT_MARKERS)
    if open_idx != -1 and (fragment_idx == -1 or open_idx <= fragment_idx):
        return open_idx
    if fragment_idx != -1:
        return _orphan_fragment_start(text, fragment_idx)
    return len(text)


def parse_dsml_tool_calls(text: str) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for match in re.finditer(re.escape(DSML_OPEN) + r"(.*?)" + re.escape(DSML_CLOSE), text, re.DOTALL):
        block = match.group(1)

        for tc in re.finditer(
            r"<name>\s*(.*?)\s*</name>.*?<parameters>\s*(.*?)\s*</parameters>",
            block,
            re.DOTALL,
        ):
            results.append(make_tool_call(tc.group(1).strip(), tc.group(2).strip()))

        invoke_pat = re.escape("<" + DSML_BAR + "DSML" + DSML_BAR + "invoke") + r'\s+name="([^"]+)"\s*>'
        for inv in re.finditer(invoke_pat, block):
            fn_name = inv.group(1)
            after_invoke = block[inv.end():]
            end_invoke = re.search(re.escape("</" + DSML_BAR + "DSML" + DSML_BAR + "invoke>"), after_invoke)
            if not end_invoke:
                continue
            param_block = after_invoke[:end_invoke.start()]
            params: Dict[str, Any] = {}
            for p in re.finditer(
                re.escape("<" + DSML_BAR + "DSML" + DSML_BAR + "parameter")
                + r'\s+name="([^"]+)"\s*(?:string="[^"]*"\s*)?'
                + r">(.*?)</" + DSML_BAR + "DSML" + DSML_BAR + "parameter>",
                param_block,
                re.DOTALL,
            ):
                params[p.group(1)] = p.group(2).strip()
            results.append(make_tool_call(fn_name, json.dumps(params, ensure_ascii=False)))
    return results


def parse_qwen_xml_tool_calls(text: str) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for match in re.finditer(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL):
        block = match.group(1)
        name_m = re.search(r"<name>\s*(.*?)\s*</name>", block, re.DOTALL)
        args_m = re.search(r"<parameters>\s*(.*?)\s*</parameters>", block, re.DOTALL)
        if name_m:
            results.append(make_tool_call(name_m.group(1).strip(), args_m.group(1).strip() if args_m else "{}"))
            continue

        func_m = re.search(r"<function=([^>]+)>(.*?)</function>", block, re.DOTALL)
        if func_m:
            fn_name = func_m.group(1).strip()
            params: Dict[str, Any] = {}
            for p in re.finditer(r"<parameter=([^>]+)>(.*?)</parameter>", func_m.group(2), re.DOTALL):
                params[p.group(1).strip()] = p.group(2).strip()
            results.append(make_tool_call(fn_name, json.dumps(params, ensure_ascii=False)))
    return results


def parse_raw_tool_calls(text: str) -> List[Dict[str, Any]]:
    parsed = parse_dsml_tool_calls(text)
    if parsed:
        return parsed
    parsed = parse_qwen_xml_tool_calls(text)
    if parsed:
        return parsed
    if has_complete_raw_tool_block(text):
        log.warning("parse_raw_tool_calls failed on: %s", text[:800])
    return []
