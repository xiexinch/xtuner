from types import SimpleNamespace

import torch

from xtuner.v1.module.rope import rope


def test_qwen3_vl_mrope_uses_yarn_initializer(monkeypatch):
    calls = []

    def fake_yarn_init(config, device=None, **kwargs):
        calls.append((config, device, kwargs))
        return torch.ones(32, device=device), 1.25

    monkeypatch.setitem(rope.ROPE_INIT_FUNCTIONS, "yarn", fake_yarn_init)
    config = SimpleNamespace(
        max_position_embeddings=1_000_000,
        rope_parameters_cfg=rope.RopeParametersConfig(
            rope_theta=10_000_000.0,
            rope_type="yarn",
            mrope_section=[11, 11, 10],
            partial_rotary_factor=0.25,
            factor=4.0,
            original_max_position_embeddings=262_144,
        ),
        rope_scaling_cfg=None,
    )

    embedding = rope.get_rope_embedding(config)

    assert isinstance(embedding, rope.Qwen3VLTextRotaryEmbedding)
    assert embedding.rope_type == "yarn"
    assert len(calls) == 1
    assert calls[0][0] is config

    hidden_states = torch.zeros(1, 4, 64)
    position_ids = torch.zeros(1, 4, dtype=torch.long)
    cos, sin = embedding(hidden_states, position_ids)

    torch.testing.assert_close(cos, torch.full_like(cos, 1.25))
    torch.testing.assert_close(sin, torch.zeros_like(sin))
