import json
import os
from pathlib import Path

import pytest

from xtuner.v1.model import Qwen3_5_VLDense27BConfig, get_model_config
from xtuner.v1.model.dense.qwen3_5_text import Qwen3_5_VLTextDense27BConfig


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
    assert (config.pad_token_id, config.eos_token_id) == (None, 248044)
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


def test_qwen3_8_checkpoint_has_only_the_deferred_mtp_parameters():
    if QWEN3_8_27B_PATH is None:
        pytest.skip("QWEN3_8_27B_PATH is required for the pinned Qwen3.8 checkpoint index")

    index_path = Path(QWEN3_8_27B_PATH) / "model.safetensors.index.json"
    with index_path.open() as file:
        weight_map = json.load(file)["weight_map"]

    mtp_keys = sorted(key for key in weight_map if key.startswith("mtp."))
    assert mtp_keys == EXPECTED_MTP_KEYS
