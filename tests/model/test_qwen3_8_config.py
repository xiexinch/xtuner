import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import transformers
from transformers import AutoConfig
from xtuner._testing import HFConfigFieldDependency, check_hf_config_save
from xtuner.v1.model import BaseModel, Qwen3_5_VLDense27BConfig, get_model_config, get_model_config_from_hf
from xtuner.v1.model.dense.qwen3_5_text import Qwen3_5_VLTextDense27BConfig
from xtuner.v1.module.rope import RopeParametersConfig


QWEN3_8_27B_PATH = os.environ.get("QWEN3_8_27B_PATH")
EXPECTED_MTP_KEYS = [
    "mtp.fc.weight",
    "mtp.layers.0.input_layernorm.weight",
    "mtp.layers.0.mlp.down_proj.weight",
    "mtp.layers.0.mlp.gate_proj.weight",
    "mtp.layers.0.mlp.up_proj.weight",
    "mtp.layers.0.post_attention_layernorm.weight",
    "mtp.layers.0.self_attn.k_norm.weight",
    "mtp.layers.0.self_attn.k_proj.weight",
    "mtp.layers.0.self_attn.o_proj.weight",
    "mtp.layers.0.self_attn.q_norm.weight",
    "mtp.layers.0.self_attn.q_proj.weight",
    "mtp.layers.0.self_attn.v_proj.weight",
    "mtp.norm.weight",
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight",
]


def test_qwen3_8_27b_text_config_matches_official_shape():
    config = Qwen3_5_VLTextDense27BConfig()

    assert (config.vocab_size, config.max_position_embeddings) == (248320, 262144)
    assert (config.pad_token_id, config.bos_token_id, config.eos_token_id) == (None, 248044, 248044)
    assert (config.num_hidden_layers, config.hidden_size, config.intermediate_size) == (64, 5120, 17408)
    assert config.tie_word_embeddings is False
    assert config.layers_type == [
        "full_attention" if (index + 1) % 4 == 0 else "linear_attention" for index in range(64)
    ]

    assert config.attention.model_dump(exclude={"attn_impl"}) == {
        "num_attention_heads": 24,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "dropout": 0.0,
        "qkv_bias": False,
        "qk_norm": True,
        "rms_norm_eps": 1e-6,
        "rms_norm_type": "zero_centered",
        "o_bias": False,
        "sliding_window": -1,
        "with_sink": False,
        "with_gate": True,
    }
    assert config.linear_attention.model_dump() == {
        "num_value_heads": 48,
        "num_key_heads": 16,
        "key_head_dim": 128,
        "value_head_dim": 128,
        "conv_kernel_dim": 4,
        "hidden_act": "silu",
        "rms_norm_eps": 1e-6,
    }
    assert config.rope_parameters_cfg.model_dump(exclude_none=True) == {
        "rope_theta": 10_000_000.0,
        "rope_type": "qwen3_vl",
        "truncate": False,
        "mrope_section": [11, 11, 10],
        "partial_rotary_factor": 0.25,
    }
    assert not any(field.startswith("mtp") for field in type(config).model_fields)


def test_qwen3_8_27b_compose_config_and_alias():
    config = Qwen3_5_VLDense27BConfig()

    assert (config.vision_config.depth, config.vision_config.hidden_size) == (27, 1152)
    assert config.vision_config.intermediate_size == 4304
    assert (config.projector_config.vision_hidden_size, config.projector_config.text_hidden_size) == (1152, 5120)
    assert isinstance(config.text_config, Qwen3_5_VLTextDense27BConfig)

    alias_config = get_model_config("QWEN3.8_VL_27B")
    assert isinstance(alias_config, Qwen3_5_VLDense27BConfig)


def test_qwen3_8_hf_path_requires_an_explicit_model_config(monkeypatch, tmp_path):
    monkeypatch.setattr(
        AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(model_type="qwen3_5"),
    )

    with pytest.raises(ValueError, match="automatic model-config detection is ambiguous"):
        get_model_config_from_hf(tmp_path)


def test_qwen3_8_native_hf_config_export(tmp_path):
    config = Qwen3_5_VLDense27BConfig()
    config.save_hf(tmp_path)

    with (tmp_path / "config.json").open() as file:
        exported = json.load(file)

    assert exported["model_type"] == "qwen3_5"
    assert exported["architectures"] == ["Qwen3_5ForConditionalGeneration"]
    assert exported["language_model_only"] is False
    assert exported["text_config"] == {
        "attention_bias": False,
        "attention_dropout": 0.0,
        "attn_output_gate": True,
        "bos_token_id": 248044,
        "dtype": "bfloat16",
        "eos_token_id": 248044,
        "full_attention_interval": 4,
        "head_dim": 256,
        "hidden_act": "silu",
        "hidden_size": 5120,
        "initializer_range": 0.02,
        "intermediate_size": 17408,
        "layer_types": ["full_attention" if (index + 1) % 4 == 0 else "linear_attention" for index in range(64)],
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 48,
        "linear_value_head_dim": 128,
        "mamba_ssm_dtype": "float32",
        "max_position_embeddings": 262144,
        "model_type": "qwen3_5_text",
        "mtp_num_hidden_layers": 1,
        "mtp_use_dedicated_embeddings": False,
        "num_attention_heads": 24,
        "num_hidden_layers": 64,
        "num_key_value_heads": 4,
        "output_gate_type": "swish",
        "pad_token_id": None,
        "partial_rotary_factor": 0.25,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {
            "mrope_interleaved": True,
            "mrope_section": [11, 11, 10],
            "partial_rotary_factor": 0.25,
            "rope_theta": 10_000_000.0,
            "rope_type": "default",
        },
        "tie_word_embeddings": False,
        "use_cache": True,
        "vocab_size": 248320,
    }
    assert exported["vision_config"] == {
        "deepstack_visual_indexes": [],
        "depth": 27,
        "hidden_act": "gelu_pytorch_tanh",
        "hidden_size": 1152,
        "in_channels": 3,
        "initializer_range": 0.02,
        "intermediate_size": 4304,
        "model_type": "qwen3_5_vision",
        "num_heads": 16,
        "num_position_embeddings": 2304,
        "out_hidden_size": 5120,
        "patch_size": 16,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
    }

    reloaded = AutoConfig.from_pretrained(tmp_path)
    assert reloaded.model_type == "qwen3_5"
    assert reloaded.text_config.max_position_embeddings == 262144


def test_qwen3_8_one_million_hf_config_export(tmp_path):
    config = Qwen3_5_VLDense27BConfig()
    config.text_config.max_position_embeddings = 1_000_000
    config.text_config.rope_parameters_cfg = RopeParametersConfig(
        rope_theta=10_000_000.0,
        rope_type="yarn",
        mrope_section=[11, 11, 10],
        partial_rotary_factor=0.25,
        factor=4.0,
        original_max_position_embeddings=262144,
    )
    config.save_hf(tmp_path)

    reloaded = AutoConfig.from_pretrained(tmp_path)
    assert reloaded.text_config.max_position_embeddings == 1_000_000
    assert reloaded.text_config.rope_parameters == {
        "factor": 4.0,
        "mrope_interleaved": True,
        "mrope_section": [11, 11, 10],
        "original_max_position_embeddings": 262144,
        "partial_rotary_factor": 0.25,
        "rope_theta": 10_000_000.0,
        "rope_type": "yarn",
    }


def test_qwen3_8_hf_writer_keeps_side_files_and_overwrites_source_config(tmp_path):
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "output"
    source_dir.mkdir()
    (source_dir / "config.json").write_text('{"stale": true}', encoding="utf-8")
    (source_dir / "generation_config.json").write_text('{"eos_token_id": [248046, 248044]}', encoding="utf-8")
    (source_dir / "chat_template.jinja").write_text("official-template", encoding="utf-8")
    (source_dir / "source-weight.safetensors").write_bytes(b"not-a-real-weight")

    config = Qwen3_5_VLDense27BConfig()
    config.text_config.max_position_embeddings = 1_000_000
    config.text_config.rope_parameters_cfg = RopeParametersConfig(
        rope_theta=10_000_000.0,
        rope_type="yarn",
        mrope_section=[11, 11, 10],
        partial_rotary_factor=0.25,
        factor=4.0,
        original_max_position_embeddings=262144,
    )
    writer = SimpleNamespace(config=config, _hf_path=source_dir)
    BaseModel._write_hf_index_and_config(
        writer,
        output_dir,
        {"model.language_model.embed_tokens.weight": "model-0001.safetensors"},
    )

    with (output_dir / "config.json").open() as file:
        exported = json.load(file)
    with (output_dir / "model.safetensors.index.json").open() as file:
        index = json.load(file)

    assert exported["text_config"]["max_position_embeddings"] == 1_000_000
    assert exported["text_config"]["rope_parameters"]["rope_type"] == "yarn"
    assert (output_dir / "generation_config.json").read_text(encoding="utf-8") == (
        '{"eos_token_id": [248046, 248044]}'
    )
    assert (output_dir / "chat_template.jinja").read_text(encoding="utf-8") == "official-template"
    assert not (output_dir / "source-weight.safetensors").exists()
    assert index == {
        "metadata": {},
        "weight_map": {"model.language_model.embed_tokens.weight": "model-0001.safetensors"},
    }


def test_qwen3_8_hf_config_matches_transformers_and_engine_contracts():
    if QWEN3_8_27B_PATH is None:
        pytest.skip("QWEN3_8_27B_PATH is required for the pinned Qwen3.8 config round-trip")

    report = check_hf_config_save(
        Qwen3_5_VLDense27BConfig(),
        QWEN3_8_27B_PATH,
        engine_dependencies=(
            HFConfigFieldDependency(
                engine="vllm",
                version="0.26.0",
                path="/text_config/layer_types/3",
                expected="full_attention",
                reason="vLLM indexes layer_types to select the decoder implementation for every layer.",
                source=(
                    "https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/model_executor/models/qwen3_5.py#L243-L251"
                ),
            ),
            HFConfigFieldDependency(
                engine="sglang",
                version="0.5.16",
                path="/text_config/max_position_embeddings",
                expected=262144,
                reason="SGLang reads this field when constructing Qwen3.5 rotary embeddings.",
                source=(
                    "https://github.com/sgl-project/sglang/blob/v0.5.16/python/sglang/srt/models/qwen3_5.py#L761-L785"
                ),
            ),
        ),
    )

    assert report.transformers_version == transformers.__version__
    assert report.checked_engine_versions == ("vllm==0.26.0", "sglang==0.5.16")


def test_qwen3_8_checkpoint_has_only_the_deferred_mtp_parameters():
    if QWEN3_8_27B_PATH is None:
        pytest.skip("QWEN3_8_27B_PATH is required for the pinned Qwen3.8 checkpoint index")

    index_path = Path(QWEN3_8_27B_PATH) / "model.safetensors.index.json"
    with index_path.open() as file:
        weight_map = json.load(file)["weight_map"]

    mtp_keys = sorted(key for key in weight_map if key.startswith("mtp."))
    assert mtp_keys == EXPECTED_MTP_KEYS
