# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer
    from xtuner.v1.data_proto.templates import HybridChatTemplate


IGNORE_INDEX = -100

_XHIGH_REASONING_INSTRUCTION = (
    "Reasoning effort is set to xhigh. Please think carefully through the task, "
    "validate key assumptions, consider plausible alternatives, and prioritize "
    "correctness, consistency, and clarity in the final answer."
)
_LOW_REASONING_INSTRUCTION = (
    "Reasoning effort is set to low. Keep your thinking brief and focused, moving "
    "directly to the conclusion without unnecessary elaboration."
)
_TOOL_SYSTEM = "# Tools\n\nYou have access to the following functions:\n\n<tools>"
_TOOL_INSTRUCTIONS = (
    "\n</tools>\n\n"
    "If you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
    "<tool_call>\n"
    "<function=example_function_name>\n"
    "<parameter=example_parameter_1>\n"
    "value_1\n"
    "</parameter>\n"
    "<parameter=example_parameter_2>\n"
    "This is the value for the second parameter\n"
    "that can span\n"
    "multiple lines\n"
    "</parameter>\n"
    "</function>\n"
    "</tool_call>\n\n"
    "<IMPORTANT>\n"
    "Reminder:\n"
    "- Function calls MUST follow the specified format: an inner <function=...></function> "
    "block must be nested within <tool_call></tool_call> XML tags\n"
    "- Required parameters MUST be specified\n"
    "- You may provide optional reasoning for your function call in natural language BEFORE "
    "the function call, but NOT after\n"
    "- If there is no function call available, answer the question like normal with your "
    "current knowledge and do not tell the user about function calls\n"
    "</IMPORTANT>"
)


def _get_offset_mapping(tokenizer, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    """Build offsets for slow tokenizers that do not expose offset mapping."""
    input_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    offsets: list[tuple[int, int]] = []
    position = 0
    pending_ids: list[int] = []

    for token_id in input_ids:
        pending_ids.append(token_id)
        decoded = tokenizer.decode(pending_ids, skip_special_tokens=False)
        if not decoded:
            continue

        start = text.find(decoded, position)
        if start >= 0:
            end = start + len(decoded)
            offsets.extend([(start, end)] * len(pending_ids))
            position = end
            pending_ids = []
        elif "\ufffd" not in decoded or len(pending_ids) >= 8:
            offsets.extend([(position, position)] * len(pending_ids))
            pending_ids = []

    offsets.extend([(position, position)] * len(pending_ids))
    return input_ids, offsets


def _render_content(
    content: Any,
    *,
    do_vision_count: bool,
    image_count: int,
    video_count: int,
    add_vision_id: bool,
    is_system_content: bool = False,
) -> tuple[str, int, int]:
    if isinstance(content, str):
        return content, image_count, video_count
    if content is None:
        return "", image_count, video_count
    if not isinstance(content, Sequence) or isinstance(content, (str, bytes, Mapping)):
        raise TypeError("Unexpected content type.")

    rendered = ""
    for item in content:
        if not isinstance(item, Mapping):
            raise TypeError("Unexpected item type in content.")

        if "image" in item or "image_url" in item or item.get("type") == "image":
            if is_system_content:
                raise ValueError("System message cannot contain images.")
            if do_vision_count:
                image_count += 1
            if add_vision_id:
                rendered += f"Picture {image_count}: "
            rendered += "<|vision_start|><|image_pad|><|vision_end|>"
        elif "video" in item or "video_url" in item or item.get("type") in ("video", "video_url"):
            if is_system_content:
                raise ValueError("System message cannot contain videos.")
            if do_vision_count:
                video_count += 1
            if add_vision_id:
                rendered += f"Video {video_count}: "
            rendered += "<|vision_start|><|video_pad|><|vision_end|>"
        elif "text" in item:
            rendered += str(item["text"])
        else:
            raise ValueError("Unexpected item type in content.")
    return rendered, image_count, video_count


def _reasoning_instruction(
    enable_thinking: bool,
    reasoning_effort: Literal["xhigh", "medium", "low"],
) -> str:
    if not enable_thinking:
        return ""
    if reasoning_effort == "xhigh":
        return _XHIGH_REASONING_INSTRUCTION
    if reasoning_effort == "medium":
        return ""
    if reasoning_effort == "low":
        return _LOW_REASONING_INSTRUCTION
    raise ValueError(
        f"Unexpected reasoning effort {reasoning_effort}. Supported types are xhigh (default), medium, and low."
    )


def _render_tool_arguments(arguments: Mapping[str, Any]) -> str:
    rendered = ""
    for name, value in arguments.items():
        rendered += f"<parameter={name}>\n"
        rendered += value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        rendered += "\n</parameter>\n"
    return rendered


def render_qwen38_chat(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    add_generation_prompt: bool = False,
    add_vision_id: bool = False,
    enable_thinking: bool | None = True,
    reasoning_effort: Literal["xhigh", "medium", "low"] = "xhigh",
    preserve_thinking: bool = True,
) -> tuple[str, list[bool]]:
    """Render the pinned Qwen3.8 template and its SFT character loss mask."""
    if not messages:
        raise ValueError("No messages provided.")
    if enable_thinking is None:
        enable_thinking = True

    text = ""
    loss_mask: list[bool] = []
    image_count = 0
    video_count = 0

    def append(value: str, loss: bool) -> None:
        nonlocal text
        text += value
        loss_mask.extend([loss] * len(value))

    def render_content(content: Any, do_vision_count: bool, is_system_content: bool = False) -> str:
        nonlocal image_count, video_count
        rendered, image_count, video_count = _render_content(
            content,
            do_vision_count=do_vision_count,
            image_count=image_count,
            video_count=video_count,
            add_vision_id=add_vision_id,
            is_system_content=is_system_content,
        )
        return rendered

    reasoning_instruction = _reasoning_instruction(enable_thinking, reasoning_effort)
    first_message = messages[0]

    if tools:
        append("<|im_start|>system\n", False)
        if reasoning_instruction:
            append(reasoning_instruction + "\n\n", False)
        append(_TOOL_SYSTEM, False)
        for tool in tools:
            append("\n" + json.dumps(tool, ensure_ascii=False), False)
        append(_TOOL_INSTRUCTIONS, False)
        if first_message.get("role") == "system":
            system_content = render_content(
                first_message.get("content"),
                False,
                is_system_content=True,
            ).strip()
            if system_content:
                append("\n\n" + system_content, False)
        append("<|im_end|>\n", False)
    elif first_message.get("role") == "system":
        system_content = render_content(
            first_message.get("content"),
            False,
            is_system_content=True,
        ).strip()
        if system_content:
            prefix = reasoning_instruction + "\n\n" if reasoning_instruction else ""
            append(f"<|im_start|>system\n{prefix}{system_content}<|im_end|>\n", False)
        elif reasoning_instruction:
            append(f"<|im_start|>system\n{reasoning_instruction}<|im_end|>\n", False)
    elif reasoning_instruction:
        append(f"<|im_start|>system\n{reasoning_instruction}<|im_end|>\n", False)

    last_query_index: int | None = None
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if message.get("role") != "user":
            continue
        content, _, _ = _render_content(
            message.get("content"),
            do_vision_count=False,
            image_count=image_count,
            video_count=video_count,
            add_vision_id=add_vision_id,
        )
        content = content.strip()
        if not (content.startswith("<tool_response>") and content.endswith("</tool_response>")):
            last_query_index = index
            break
    if last_query_index is None:
        raise ValueError("No user query found in messages.")

    for index, message in enumerate(messages):
        role = message.get("role")
        content = render_content(message.get("content"), True).strip()

        if role == "system":
            if index != 0:
                raise ValueError("System message must be at the beginning.")
            continue

        if role == "user":
            append(f"<|im_start|>user\n{content}<|im_end|>\n", False)
            continue

        if role == "assistant":
            assistant_loss = bool(message.get("loss", True))
            reasoning_content = message.get("reasoning_content")
            reasoning_content = reasoning_content.strip() if isinstance(reasoning_content, str) else ""
            render_thinking = preserve_thinking or index > last_query_index

            append("<|im_start|>assistant\n", False)
            if render_thinking:
                append("<think>\n", False)
                if reasoning_content:
                    append(reasoning_content + "\n</think>\n\n", assistant_loss)
                elif enable_thinking:
                    append("\n</think>\n\n", assistant_loss)
                else:
                    # With thinking disabled, the official generation prompt already
                    # contains this empty block, so it remains prompt scaffolding.
                    append("\n</think>\n\n", False)
            append(content, assistant_loss)

            tool_calls = message.get("tool_calls")
            if tool_calls:
                if isinstance(tool_calls, Mapping) or not isinstance(tool_calls, Sequence):
                    raise TypeError("assistant tool_calls must be a sequence.")
                for tool_index, raw_tool_call in enumerate(tool_calls):
                    tool_call = raw_tool_call.get("function", raw_tool_call)
                    if tool_index == 0:
                        if content:
                            append("\n\n", assistant_loss)
                        append(f"<tool_call>\n<function={tool_call['name']}>\n", assistant_loss)
                    else:
                        append(f"\n<tool_call>\n<function={tool_call['name']}>\n", assistant_loss)

                    arguments = tool_call.get("arguments")
                    if arguments is not None and arguments != "":
                        if not isinstance(arguments, Mapping):
                            raise TypeError("tool call arguments must be a mapping.")
                        append(_render_tool_arguments(arguments), assistant_loss)
                    append("</function>\n</tool_call>", assistant_loss)
            append("<|im_end|>\n", assistant_loss)
            continue

        if role == "tool":
            previous_role = messages[index - 1].get("role") if index > 0 else None
            if previous_role != "tool":
                append("<|im_start|>user", False)
            append("\n<tool_response>\n", False)
            append(content, False)
            append("\n</tool_response>", False)
            next_role = messages[index + 1].get("role") if index + 1 < len(messages) else None
            if next_role != "tool":
                append("<|im_end|>\n", False)
            continue

        raise ValueError("Unexpected message role.")

    if add_generation_prompt:
        append("<|im_start|>assistant\n", False)
        if enable_thinking:
            append("<think>\n", False)
        else:
            append("<think>\n\n</think>\n\n", False)

    return text, loss_mask


def _tokenize_with_loss_mask(
    tokenizer: PreTrainedTokenizer,
    text: str,
    loss_mask: list[bool],
) -> tuple[list[int], list[int]]:
    try:
        encoded = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        input_ids = encoded["input_ids"]
        offsets = encoded["offset_mapping"]
    except Exception:
        input_ids, offsets = _get_offset_mapping(tokenizer, text)

    labels = []
    for token_id, (start, end) in zip(input_ids, offsets):
        if start == end or not any(loss_mask[start:end]):
            labels.append(IGNORE_INDEX)
        else:
            labels.append(token_id)
    return input_ids, labels


def qwen38_tokenize_fn_fastspeed(
    tokenizer: PreTrainedTokenizer,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    add_generation_prompt: bool = False,
    add_vision_id: bool = False,
    enable_thinking: bool | None = True,
    reasoning_effort: Literal["xhigh", "medium", "low"] = "xhigh",
    preserve_thinking: bool = True,
    **kwargs,
) -> tuple[list[int], list[int]]:
    text, loss_mask = render_qwen38_chat(
        messages,
        tools=tools,
        add_generation_prompt=add_generation_prompt,
        add_vision_id=add_vision_id,
        enable_thinking=enable_thinking,
        reasoning_effort=reasoning_effort,
        preserve_thinking=preserve_thinking,
    )
    return _tokenize_with_loss_mask(tokenizer, text, loss_mask)


class Qwen38ChatMessages(BaseModel):
    model_config = ConfigDict(extra="forbid")
    messages: List[Dict[str, Any]]
    tools: Optional[List[Dict[str, Any]]] = None

    def tokenize(
        self,
        tokenizer: PreTrainedTokenizer,
        chat_template: HybridChatTemplate | None = None,
        add_vision_id: bool = False,
        add_generation_prompt: bool = False,
        enable_thinking: bool | None = True,
        reasoning_effort: Literal["xhigh", "medium", "low"] = "xhigh",
        preserve_thinking: bool = True,
        **kwargs,
    ) -> Dict[str, list[int]]:
        if len(self.messages) == 1 and self.messages[0].get("role") == "pretrain":
            text, _, _ = _render_content(
                self.messages[0].get("content"),
                do_vision_count=True,
                image_count=0,
                video_count=0,
                add_vision_id=add_vision_id,
            )
            input_ids = tokenizer.encode(text, add_special_tokens=False)
            return {"input_ids": input_ids, "labels": copy.deepcopy(input_ids)}

        messages = copy.deepcopy(self.messages)
        if chat_template is not None and chat_template.default_system is not None:
            if messages[0].get("role") == "system":
                messages[0]["content"] = chat_template.default_system
            else:
                messages.insert(0, {"role": "system", "content": chat_template.default_system})

        input_ids, labels = qwen38_tokenize_fn_fastspeed(
            tokenizer,
            messages,
            tools=self.tools,
            add_generation_prompt=add_generation_prompt,
            add_vision_id=add_vision_id,
            enable_thinking=enable_thinking,
            reasoning_effort=reasoning_effort,
            preserve_thinking=preserve_thinking,
            **kwargs,
        )
        return {"input_ids": input_ids, "labels": labels}
