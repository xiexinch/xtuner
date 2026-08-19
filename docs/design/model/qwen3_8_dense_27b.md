# Qwen3.8-27B dense VLM integration design

## Reference

- Model: [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0)
- Immutable revision: `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
- Hugging Face architecture: `Qwen3_5ForConditionalGeneration`
- Hugging Face model type: `qwen3_5`
- Validation tokenizer: `Qwen2Tokenizer` loaded with Transformers 5.14.1

Qwen3.8 keeps the Qwen3.5 implementation names and checkpoint layout. The
integration therefore extends the existing `Qwen3_5_*` XTuner family instead of
adding a parallel model body.

## Model configuration

The text tower is a 64-layer dense hybrid: three GatedDeltaNet layers followed
by one gated full-attention layer in every group of four. It reuses
`Qwen3_5_VLTextDenseConfig` and adds only the published 27B dimensions:

- hidden size 5120, intermediate size 17408;
- 24 query heads, 4 KV heads, head dimension 256;
- 48 linear-attention value heads and 16 key heads;
- untied embeddings;
- interleaved partial mRoPE with section `[11, 11, 10]` and theta `1e7`.

The compose config reuses `Qwen3_5_BaseConfig`, with a depth-27, hidden-1152
vision tower and a 1152-to-5120 projector. The public alias is
`qwen3.8-vl-27b`. This integration supports **explicit config selection only**:
use `Qwen3_5_VLDense27BConfig()` (as the SFT example does) or the
`qwen3.8-vl-27b` alias. `get_model_config_from_hf` intentionally does not guess
between Qwen3.5 and Qwen3.8 because both publish `model_type=qwen3_5`.

## Chat rendering and loss

`qwen3.8-vl` gets a dedicated renderer rather than changing
`qwen3.5-vl`. It mirrors the official `chat_template.jinja` at the pinned
revision, including:

- `enable_thinking=True`, `reasoning_effort="xhigh"`, and
  `preserve_thinking=True` by default;
- the exact xhigh/low system instructions and no instruction for medium;
- historical `reasoning_content` preservation by default, without parsing
  `<think>` from `content`;
- raw string tool-argument values and JSON serialization for other values;
- the Qwen3.8 tool and multimodal branches.

The dedicated renderer also accepts XTuner's internal `video_url` content item
as an alias for the official template's `video` item, so existing
`VLMJsonlDataset` video annotations reach the same serialized video placeholder.

Assistant role scaffolding and generation prompts are masked. Demonstrated
assistant reasoning, content, tool calls, and the `<|im_end|>` boundary receive
loss by default. `loss=False` masks the complete assistant output, including
its termination boundary.

### Stop contract

The tokenizer and model metadata expose two relevant special tokens:

- tokenizer EOS: `<|im_end|>` / `248046`;
- text-config EOS: `<|endoftext|>` / `248044`;
- generation-config EOS list: `[248046, 248044]`.

The chat template itself always terminates a demonstrated assistant message
with `<|im_end|>\n` (`[248046, 198]`). There is no extra SFT-only suffix.

| Transition | Serialized boundary | Token IDs | Engine mechanism | `loss=True` | `loss=False` |
|---|---|---|---|---|---|
| assistant to user | `<|im_end|>\n` | `[248046, 198]` | EOS/stop ID `248046` | both tokens supervised | both tokens masked |
| assistant tool call to tool result | `<|im_end|>\n` | `[248046, 198]` | EOS/stop ID `248046` | both tokens supervised | both tokens masked |
| final demonstrated assistant to end | `<|im_end|>\n` | `[248046, 198]` | EOS/stop ID `248046` | both tokens supervised | both tokens masked |

The XTuner template advertises both `<|im_end|>` and `<|endoftext|>` as stop
words, matching the official generation metadata. Serving may stop before the
serialized newline, while SFT deliberately trains that newline as part of the
assistant boundary.

## MTP decision

The checkpoint contains one dense MTP layer represented by 15 `mtp.*` keys.
Dense MTP remains deferred, matching the existing Qwen3.5 dense integration.
The compose loader calls each tower with `strict=False`, so those checkpoint-only
keys are ignored while every expected text, vision, and projector parameter is
still required. XTuner does not emit `mtp.*` keys when saving.

Implementing dense MTP is a separate change because it requires model-body,
forward/loss, checkpoint-mapping, and parity work rather than a size-config
extension.

## Hugging Face save and serving contract

`Qwen3_5_BaseConfig` continues to use the compose family's existing
`hf_config=None` path. A model loaded from Hugging Face retains `_hf_path`, and
`save_hf` copies the source `config.json`, tokenizer, processor, and template
side files before replacing the weight index. Consequently, a saved checkpoint
keeps the official `model_type=qwen3_5`,
`architectures=["Qwen3_5ForConditionalGeneration"]`, nested text/vision fields,
token IDs, and generation metadata. Config-only construction still cannot emit
a standalone Hugging Face config; that is an existing family-level limitation,
not silently approximated by this size extension.

The latest stable serving releases available during this integration were
audited at exact tags (source audit only; no GPU engine smoke test):

| Engine | Version | Static contract |
|---|---:|---|
| vLLM | [0.26.0](https://pypi.org/project/vllm/0.26.0/) | [Registers](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/model_executor/models/registry.py#L560) `Qwen3_5ForConditionalGeneration` and its [dense/VL loader](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/model_executor/models/qwen3_5.py) skips `mtp.*`. The [Hugging Face renderer](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/renderers/hf.py) delegates to `tokenizer.apply_chat_template`, and [sampling](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/sampling_params.py#L598-L624) merges generation-config EOS IDs with the tokenizer EOS. |
| SGLang | [0.5.16](https://pypi.org/project/sglang/0.5.16/) | [Exports](https://github.com/sgl-project/sglang/blob/v0.5.16/python/sglang/srt/models/qwen3_5.py#L2045) `Qwen3_5ForConditionalGeneration`; its [dense/VL loader](https://github.com/sgl-project/sglang/blob/v0.5.16/python/sglang/srt/models/qwen3_5.py#L1591-L1611) skips names containing `mtp`. Its [chat endpoint](https://github.com/sgl-project/sglang/blob/v0.5.16/python/sglang/srt/entrypoints/openai/serving_chat.py#L951-L994) delegates to `apply_chat_template`, and [model config](https://github.com/sgl-project/sglang/blob/v0.5.16/python/sglang/srt/configs/model_config.py#L1411-L1429) unions the generation-config EOS IDs. |

These engines therefore consume the unchanged official source config, render
the official template, honor both published EOS IDs, and apply the same current
MTP deferral. Static inspection proves the configuration, rendering, stop, and
weight-loading contracts, but not end-to-end runtime compatibility on a
specific engine build or accelerator.

## Verification

- Assert all 27B config fields and the public alias.
- Assert the official checkpoint index contains exactly the documented 15 MTP
  keys and that the 27B XTuner model has no MTP parameters.
- Compare rendered text and token IDs against the pinned official tokenizer for
  default thinking, reasoning-effort variants, historical thinking, tools,
  multimodal content, generation prompts, and `loss=False`.
- Assert assistant output and `<|im_end|>` label ownership directly.
