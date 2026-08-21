"""Qwen3.8 rendering and loss-mask parity against the pinned HF template.

Reference: Qwen/Qwen3.8-27B at
1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0.
"""

import os

import pytest

from transformers import AutoTokenizer
from xtuner.v1.data_proto.messages.qwen38_chat import render_qwen38_chat
from xtuner.v1.datasets import OpenaiTokenizeFunctionConfig
from xtuner.v1.datasets.mllm_tokenize_fn import qwen3_vl_tokenize_fn


QWEN3_8_27B_PATH = os.environ.get("QWEN3_8_27B_PATH")
IGNORE_INDEX = -100


@pytest.fixture(scope="module")
def tokenizer():
    if QWEN3_8_27B_PATH is None:
        pytest.skip("QWEN3_8_27B_PATH is required for the pinned Qwen3.8 tokenizer oracle")
    return AutoTokenizer.from_pretrained(QWEN3_8_27B_PATH)


@pytest.fixture(scope="module")
def tokenize_fn(tokenizer):
    return OpenaiTokenizeFunctionConfig(chat_template="qwen3.8-vl").build(tokenizer)


def _hf_render(tokenizer, messages, tools=None, **kwargs):
    return tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        **kwargs,
    )


def _assert_render_parity(tokenizer, tokenized, messages, tools=None, **kwargs):
    expected_text = _hf_render(tokenizer, messages, tools=tools, **kwargs)
    expected_ids = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=True,
        **kwargs,
    )
    if hasattr(expected_ids, "keys"):
        expected_ids = expected_ids["input_ids"]

    assert tokenized["input_ids"] == expected_ids
    assert tokenizer.decode(tokenized["input_ids"], skip_special_tokens=False) == expected_text
    assert len(tokenized["input_ids"]) == len(tokenized["labels"])
    assert all(
        label == IGNORE_INDEX or label == token_id
        for token_id, label in zip(tokenized["input_ids"], tokenized["labels"])
    )
    return expected_text


def _find_subsequence(values, subsequence, start=0):
    for index in range(start, len(values) - len(subsequence) + 1):
        if values[index : index + len(subsequence)] == subsequence:
            return index
    raise AssertionError(f"subsequence not found: {subsequence}")


def _assert_token_span(tokenizer, tokenized, text, supervised, start=0):
    span_ids = tokenizer.encode(text, add_special_tokens=False)
    index = _find_subsequence(tokenized["input_ids"], span_ids, start=start)
    expected = span_ids if supervised else [IGNORE_INDEX] * len(span_ids)
    assert tokenized["labels"][index : index + len(span_ids)] == expected
    return index + len(span_ids)


def test_xtuner_video_url_content_schema_is_supported():
    """The MLLM dataset protocol uses video_url, while HF uses video."""
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video_url",
                    "video_url": {"url": "ignored-by-chat-renderer.mp4"},
                },
                {"type": "text", "text": "Describe the video."},
            ],
        },
        {"role": "assistant", "content": "A short video."},
    ]

    rendered, loss_mask = render_qwen38_chat(messages)

    assert ("<|im_start|>user\n<|vision_start|><|video_pad|><|vision_end|>Describe the video.<|im_end|>\n") in rendered
    assert len(rendered) == len(loss_mask)


def test_nonleading_system_message_is_strict_by_default():
    messages = [
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "First answer"},
        {"role": "system", "content": "Updated constraint"},
        {"role": "user", "content": "Second question"},
    ]

    with pytest.raises(ValueError, match="System message must be at the beginning"):
        render_qwen38_chat(messages)


def test_glm52_data_compat_renders_nonleading_system_as_masked_qwen_turn():
    messages = [
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "First answer"},
        {"role": "system", "content": "Updated constraint"},
        {"role": "user", "content": "Second question"},
        {"role": "assistant", "content": "Second answer"},
    ]

    rendered, loss_mask = render_qwen38_chat(messages, glm52_data_compat=True)

    system_turn = "<|im_start|>system\nUpdated constraint<|im_end|>\n"
    system_start = rendered.index(system_turn)
    system_end = system_start + len(system_turn)
    final_answer = "Second answer<|im_end|>\n"
    answer_start = rendered.index(final_answer)
    answer_end = answer_start + len(final_answer)

    assert system_turn in rendered
    assert not any(loss_mask[system_start:system_end])
    assert all(loss_mask[answer_start:answer_end])
    assert len(rendered) == len(loss_mask)


def test_qwen3_vl_config_forwards_glm52_data_compat(monkeypatch):
    captured = {}

    def fake_tokenize_function(*args, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(qwen3_vl_tokenize_fn, "Qwen3VLTokenizeFunction", fake_tokenize_function)
    config = qwen3_vl_tokenize_fn.Qwen3VLTokenizeFnConfig(
        processor_path="unused-in-test",
        chat_template="qwen3.8-vl",
        glm52_data_compat=True,
    )

    config.build(object(), tokenizer_hash="fixed-tokenizer-hash", anno_name="test-data")

    assert captured["glm52_data_compat"] is True


def test_openai_config_enables_glm52_data_compat(tokenizer):
    messages = [
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "First answer"},
        {"role": "system", "content": "Updated constraint"},
        {"role": "user", "content": "Second question"},
        {"role": "assistant", "content": "Second answer"},
    ]
    strict_fn = OpenaiTokenizeFunctionConfig(chat_template="qwen3.8-vl").build(
        tokenizer,
        tokenizer_hash="fixed-tokenizer-hash",
    )
    compat_fn = OpenaiTokenizeFunctionConfig(
        chat_template="qwen3.8-vl",
        glm52_data_compat=True,
    ).build(tokenizer, tokenizer_hash="fixed-tokenizer-hash")

    with pytest.raises(ValueError, match="System message must be at the beginning"):
        strict_fn({"messages": messages})
    with pytest.raises(Exception, match="System message must be at the beginning"):
        _hf_render(tokenizer, messages)

    tokenized = compat_fn({"messages": messages})
    rendered = tokenizer.decode(tokenized["input_ids"], skip_special_tokens=False)

    assert compat_fn.hash() != strict_fn.hash()
    _assert_token_span(
        tokenizer,
        tokenized,
        "<|im_start|>system\nUpdated constraint<|im_end|>\n",
        False,
    )
    _assert_token_span(tokenizer, tokenized, "Second answer<|im_end|>\n", True)
    assert rendered.count("Updated constraint") == 1


def test_default_preserves_all_reasoning_and_supervises_each_assistant(tokenizer, tokenize_fn):
    messages = [
        {"role": "user", "content": "Repeat λ."},
        {
            "role": "assistant",
            "reasoning_content": "old trace λ",
            "content": "Repeated λ.",
        },
        {"role": "user", "content": "Repeat λ again."},
        {
            "role": "assistant",
            "reasoning_content": "new trace λ",
            "content": "Repeated λ.",
        },
    ]

    tokenized = tokenize_fn({"messages": messages})
    rendered = _assert_render_parity(tokenizer, tokenized, messages)

    assert rendered.startswith("<|im_start|>system\nReasoning effort is set to xhigh.")
    assert rendered.count("<think>") == 2
    assert "old trace λ" in rendered
    assert "new trace λ" in rendered

    cursor = _assert_token_span(tokenizer, tokenized, "old trace λ\n</think>\n\nRepeated λ.<|im_end|>\n", True)
    _assert_token_span(
        tokenizer,
        tokenized,
        "new trace λ\n</think>\n\nRepeated λ.<|im_end|>\n",
        True,
        start=cursor,
    )

    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    end_labels = [
        label for token_id, label in zip(tokenized["input_ids"], tokenized["labels"]) if token_id == im_end_id
    ]
    assert end_labels == [IGNORE_INDEX, IGNORE_INDEX, im_end_id, IGNORE_INDEX, im_end_id]


@pytest.mark.parametrize(
    ("reasoning_effort", "expected_instruction"),
    [
        ("xhigh", "Reasoning effort is set to xhigh."),
        ("medium", None),
        ("low", "Reasoning effort is set to low."),
    ],
)
def test_reasoning_effort_matches_official_system_injection(
    tokenizer,
    tokenize_fn,
    reasoning_effort,
    expected_instruction,
):
    messages = [
        {"role": "user", "content": "Question"},
        {"role": "assistant", "content": "Answer"},
    ]

    tokenized = tokenize_fn({"messages": messages}, reasoning_effort=reasoning_effort)
    rendered = _assert_render_parity(
        tokenizer,
        tokenized,
        messages,
        reasoning_effort=reasoning_effort,
    )

    if expected_instruction is None:
        assert not rendered.startswith("<|im_start|>system")
    else:
        assert expected_instruction in rendered


def test_preserve_thinking_false_only_removes_historical_trace(tokenizer, tokenize_fn):
    messages = [
        {"role": "user", "content": "First"},
        {"role": "assistant", "reasoning_content": "old trace", "content": "Old answer"},
        {"role": "user", "content": "Second"},
        {"role": "assistant", "reasoning_content": "new trace", "content": "New answer"},
    ]

    tokenized = tokenize_fn({"messages": messages}, preserve_thinking=False)
    rendered = _assert_render_parity(
        tokenizer,
        tokenized,
        messages,
        preserve_thinking=False,
    )

    assert "old trace" not in rendered
    assert "<|im_start|>assistant\nOld answer" in rendered
    assert "new trace" in rendered
    _assert_token_span(tokenizer, tokenized, "Old answer<|im_end|>\n", True)
    _assert_token_span(tokenizer, tokenized, "new trace\n</think>\n\nNew answer<|im_end|>\n", True)


def test_content_think_markup_is_not_reinterpreted_as_reasoning(tokenizer, tokenize_fn):
    messages = [
        {"role": "user", "content": "Question"},
        {"role": "assistant", "content": "<think>literal</think>Visible"},
    ]

    tokenized = tokenize_fn({"messages": messages})
    rendered = _assert_render_parity(tokenizer, tokenized, messages)

    assert "<think>\n\n</think>\n\n<think>literal</think>Visible" in rendered


def test_tools_multimodal_content_and_argument_serialization_match_hf(tokenizer, tokenize_fn):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "inspect",
                "description": "Inspect an object.",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
            },
        }
    ]
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "ignored-by-renderer"},
                {"type": "video", "video": "ignored-by-renderer"},
                {"type": "text", "text": "Inspect this."},
            ],
        },
        {
            "role": "assistant",
            "reasoning_content": "use the tool",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "inspect",
                        "arguments": {
                            "query": "raw <& é",
                            "options": {"z": 1, "a": "é"},
                            "enabled": True,
                        },
                    }
                }
            ],
        },
        {"role": "tool", "content": "tool result"},
        {"role": "assistant", "content": "Done."},
    ]

    tokenized = tokenize_fn({"messages": messages, "tools": tools})
    rendered = _assert_render_parity(tokenizer, tokenized, messages, tools=tools)

    assert (
        "<|vision_start|><|image_pad|><|vision_end|><|vision_start|><|video_pad|><|vision_end|>Inspect this."
    ) in rendered
    assert "<parameter=query>\nraw <& é\n</parameter>" in rendered
    assert '<parameter=options>\n{"z": 1, "a": "é"}\n</parameter>' in rendered
    assert "<parameter=enabled>\ntrue\n</parameter>" in rendered
    assert "<tool_response>\ntool result\n</tool_response>" in rendered
    _assert_token_span(tokenizer, tokenized, "use the tool\n</think>\n\n<tool_call>", True)
    _assert_token_span(tokenizer, tokenized, "tool result", False)


def test_loss_false_masks_output_and_termination_token(tokenizer, tokenize_fn):
    messages = [
        {"role": "user", "content": "First"},
        {"role": "assistant", "reasoning_content": "kept", "content": "Supervised"},
        {"role": "user", "content": "Second"},
        {
            "role": "assistant",
            "reasoning_content": "hidden",
            "content": "Unsupervised",
            "loss": False,
        },
    ]

    tokenized = tokenize_fn({"messages": messages})
    _assert_render_parity(tokenizer, tokenized, messages)

    _assert_token_span(tokenizer, tokenized, "kept\n</think>\n\nSupervised<|im_end|>\n", True)
    _assert_token_span(tokenizer, tokenized, "hidden\n</think>\n\nUnsupervised<|im_end|>\n", False)

    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    end_labels = [
        label for token_id, label in zip(tokenized["input_ids"], tokenized["labels"]) if token_id == im_end_id
    ]
    assert end_labels[-1] == IGNORE_INDEX


@pytest.mark.parametrize("enable_thinking", [True, False])
def test_generation_prompt_is_official_and_fully_masked(tokenizer, tokenize_fn, enable_thinking):
    messages = [{"role": "user", "content": "Question"}]

    tokenized = tokenize_fn(
        {"messages": messages},
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    rendered = _assert_render_parity(
        tokenizer,
        tokenized,
        messages,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )

    expected_suffix = (
        "<|im_start|>assistant\n<think>\n" if enable_thinking else ("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    )
    assert rendered.endswith(expected_suffix)
    assert all(label == IGNORE_INDEX for label in tokenized["labels"])
