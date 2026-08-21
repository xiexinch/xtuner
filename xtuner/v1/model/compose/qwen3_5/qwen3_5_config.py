from transformers.models.qwen3_5.configuration_qwen3_5 import (
    Qwen3_5Config as HFQwen3_5Config,
)
from transformers.models.qwen3_5.configuration_qwen3_5 import (
    Qwen3_5TextConfig as HFQwen3_5TextConfig,
)
from transformers.models.qwen3_5.configuration_qwen3_5 import (
    Qwen3_5VisionConfig as HFQwen3_5VisionConfig,
)
from xtuner.v1.model.dense.qwen3_5_text import (
    Qwen3_5_VLTextDense4BConfig,
    Qwen3_5_VLTextDense27BConfig,
    Qwen3_5_VLTextDenseConfig,
)
from xtuner.v1.model.moe.qwen3_5_text import Qwen3_5_VLTextMoE35BA3BConfig, Qwen3_5_VLTextMoEConfig
from xtuner.v1.model.moe.qwen3_5_text_split import (
    Qwen3_5_VLTextMoE35BA3BSplitConfig,
    Qwen3_5_VLTextMoE397BA17BSplitConfig,
    Qwen3_5_VLTextMoESplitConfig,
)
from xtuner.v1.utils import get_logger

from ..qwen3_vl.qwen3_vl_config import Qwen3VLBaseConfig, Qwen3VLProjectorConfig, Qwen3VLVisionConfig


logger = get_logger()


class Qwen3_5_VisionConfig(Qwen3VLVisionConfig):
    deepstack_visual_indexes: list[int] = []


class Qwen3_5_ProjectorConfig(Qwen3VLProjectorConfig):
    deepstack_visual_indexes: list[int] = []


Qwen3_5TextConfig = Qwen3_5_VLTextDenseConfig | Qwen3_5_VLTextMoEConfig | Qwen3_5_VLTextMoESplitConfig


class Qwen3_5_BaseConfig(Qwen3VLBaseConfig):
    vision_config: Qwen3_5_VisionConfig
    projector_config: Qwen3_5_ProjectorConfig
    text_config: Qwen3_5TextConfig

    image_token_id: int = 248056
    video_token_id: int = 248057
    vision_start_token_id: int = 248053
    vision_end_token_id: int = 248054


class Qwen3_5_VLMoE35BA3Config(Qwen3_5_BaseConfig):
    vision_config: Qwen3_5_VisionConfig = Qwen3_5_VisionConfig()
    projector_config: Qwen3_5_ProjectorConfig = Qwen3_5_ProjectorConfig()
    text_config: Qwen3_5_VLTextMoE35BA3BConfig = Qwen3_5_VLTextMoE35BA3BConfig()


class Qwen3_5_VLDense4BConfig(Qwen3_5_BaseConfig):
    vision_config: Qwen3_5_VisionConfig = Qwen3_5_VisionConfig(depth=24, hidden_size=1024, intermediate_size=4096)
    projector_config: Qwen3_5_ProjectorConfig = Qwen3_5_ProjectorConfig(vision_hidden_size=1024, text_hidden_size=2560)
    text_config: Qwen3_5_VLTextDense4BConfig = Qwen3_5_VLTextDense4BConfig()


class Qwen3_5_VLDense27BConfig(Qwen3_5_BaseConfig):
    """Explicit config for Qwen3.8-27B.

    Qwen3.8 and Qwen3.5 checkpoints both advertise ``model_type=qwen3_5``;
    callers must therefore select this size config explicitly instead of using
    ``get_model_config_from_hf``.
    """

    vision_config: Qwen3_5_VisionConfig = Qwen3_5_VisionConfig(
        depth=27,
        hidden_size=1152,
        intermediate_size=4304,
    )
    projector_config: Qwen3_5_ProjectorConfig = Qwen3_5_ProjectorConfig(
        vision_hidden_size=1152,
        text_hidden_size=5120,
    )
    text_config: Qwen3_5_VLTextDense27BConfig = Qwen3_5_VLTextDense27BConfig()

    def _hf_rope_parameters(self) -> dict:
        rope_config = self.text_config.rope_parameters_cfg
        if rope_config is None or rope_config.mrope_section is None:
            raise ValueError("Qwen3.8-27B HF export requires an mRoPE configuration")

        # XTuner's ``qwen3_vl`` value identifies interleaved mRoPE internally.
        # Transformers represents the same native RoPE as ``rope_type=default``
        # plus ``mrope_interleaved=true``. Keep only non-default optional values
        # so a native export round-trips exactly through Transformers 5.14.1.
        rope_parameters = rope_config.model_dump(exclude_none=True, exclude_defaults=True)
        rope_parameters.update(
            rope_theta=rope_config.rope_theta,
            rope_type="default" if rope_config.rope_type == "qwen3_vl" else rope_config.rope_type,
            mrope_interleaved=True,
            mrope_section=rope_config.mrope_section,
            partial_rotary_factor=rope_config.partial_rotary_factor,
        )
        return rope_parameters

    @property
    def hf_config(self) -> HFQwen3_5Config:
        """Build the official HF config, including runtime context changes."""
        attention = self.text_config.attention
        linear_attention = self.text_config.linear_attention
        if linear_attention is None:
            raise ValueError("Qwen3.8-27B HF export requires GatedDeltaNet parameters")

        text_config = HFQwen3_5TextConfig(
            vocab_size=self.text_config.vocab_size,
            hidden_size=self.text_config.hidden_size,
            intermediate_size=self.text_config.intermediate_size,
            num_hidden_layers=self.text_config.num_hidden_layers,
            num_attention_heads=attention.num_attention_heads,
            num_key_value_heads=attention.num_key_value_heads,
            hidden_act=self.text_config.hidden_act,
            max_position_embeddings=self.text_config.max_position_embeddings,
            initializer_range=0.02,
            rms_norm_eps=self.text_config.rms_norm_eps,
            use_cache=True,
            tie_word_embeddings=self.text_config.tie_word_embeddings,
            rope_parameters=self._hf_rope_parameters(),
            attention_bias=attention.qkv_bias or attention.o_bias,
            attention_dropout=attention.dropout,
            head_dim=attention.head_dim,
            linear_conv_kernel_dim=linear_attention.conv_kernel_dim,
            linear_key_head_dim=linear_attention.key_head_dim,
            linear_value_head_dim=linear_attention.value_head_dim,
            linear_num_key_heads=linear_attention.num_key_heads,
            linear_num_value_heads=linear_attention.num_value_heads,
            layer_types=self.text_config.layers_type,
            pad_token_id=self.text_config.pad_token_id,
            bos_token_id=self.text_config.bos_token_id,
            eos_token_id=self.text_config.eos_token_id,
            dtype="bfloat16",
            attn_output_gate=attention.with_gate,
            output_gate_type="swish",
            mamba_ssm_dtype="float32",
            # Preserve the official metadata. XTuner, Transformers, vLLM, and
            # SGLang still skip the checkpoint-only ``mtp.*`` tensors here.
            mtp_num_hidden_layers=1,
            mtp_use_dedicated_embeddings=False,
            partial_rotary_factor=self.text_config.rope_parameters_cfg.partial_rotary_factor,
        )
        # Qwen3_5TextConfig consumes this compatibility argument while deriving
        # layer_types. Restore it because the official config retains the field.
        text_config.full_attention_interval = 4

        vision_config = HFQwen3_5VisionConfig(
            depth=self.vision_config.depth,
            hidden_size=self.vision_config.hidden_size,
            hidden_act=self.vision_config.hidden_act,
            intermediate_size=self.vision_config.intermediate_size,
            num_heads=self.vision_config.num_attention_heads,
            in_channels=self.vision_config.in_channels,
            patch_size=self.vision_config.patch_size,
            spatial_merge_size=self.vision_config.spatial_merge_size,
            temporal_patch_size=self.vision_config.temporal_patch_size,
            out_hidden_size=self.projector_config.text_hidden_size,
            num_position_embeddings=self.vision_config.num_position_embeddings,
            initializer_range=self.vision_config.initializer_range,
            deepstack_visual_indexes=self.vision_config.deepstack_visual_indexes,
        )

        return HFQwen3_5Config(
            architectures=["Qwen3_5ForConditionalGeneration"],
            text_config=text_config,
            vision_config=vision_config,
            image_token_id=self.image_token_id,
            video_token_id=self.video_token_id,
            vision_start_token_id=self.vision_start_token_id,
            vision_end_token_id=self.vision_end_token_id,
            tie_word_embeddings=self.text_config.tie_word_embeddings,
            language_model_only=False,
        )


class Qwen3_5_VLMoE35BA3SplitConfig(Qwen3_5_BaseConfig):
    vision_config: Qwen3_5_VisionConfig = Qwen3_5_VisionConfig()
    projector_config: Qwen3_5_ProjectorConfig = Qwen3_5_ProjectorConfig()
    text_config: Qwen3_5_VLTextMoE35BA3BSplitConfig = Qwen3_5_VLTextMoE35BA3BSplitConfig(
        hf_key_mapping={r"^model\.": "model.language_model."}
    )


class Qwen3_5TimeSeriesMoE35BA3Config(Qwen3_5_VLMoE35BA3Config):
    time_series_encoder_path: str | None = None
    ts_token_id: int = 248093


class Qwen3_5_VLMoE397BA17SplitConfig(Qwen3_5_BaseConfig):
    vision_config: Qwen3_5_VisionConfig = Qwen3_5_VisionConfig(fully_shard=False)
    projector_config: Qwen3_5_ProjectorConfig = Qwen3_5_ProjectorConfig(text_hidden_size=4096, fully_shard=False)
    text_config: Qwen3_5_VLTextMoE397BA17BSplitConfig = Qwen3_5_VLTextMoE397BA17BSplitConfig(
        hf_key_mapping={r"^model\.": "model.language_model."}
    )
