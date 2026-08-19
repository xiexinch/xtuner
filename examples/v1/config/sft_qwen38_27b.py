"""Qwen3.8-27B dense/VL SFT example configuration.

Required environment variables:
    META_DATA_PATH
    MODEL_PATH
    WORK_DIR
    TOKENIZER_CACHE_DIR

The checkpoint at MODEL_PATH must be a complete Hugging Face Qwen3.8-27B
directory so that XTuner can load its model weights, tokenizer, processor, and
chat-template artifacts.

Qwen3.8 model-config selection is explicit because its Hugging Face
``model_type=qwen3_5`` is shared with Qwen3.5. This example therefore constructs
``Qwen3_5_VLDense27BConfig`` directly.

Set SAMPLE_MAX_LENGTH=PACK_MAX_LENGTH=MAX_POSITION_EMBEDDINGS=1000000 and
ROPE_SCALING_FACTOR=4.0 to use the official static-YaRN 1M extension.
"""

import json
import os
import shutil
from pathlib import Path
from typing import Any

from xtuner.v1.config import FSDPConfig, LRConfig, MuonConfig
from xtuner.v1.datasets import (
    PretrainTokenizeFunctionConfig,
    Qwen3VLTokenizeFnConfig,
)
from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfig
from xtuner.v1.datasets.mllm_tokenize_fn import OSSLoaderConfig
from xtuner.v1.float8.config import Float8Config, ScalingGranularity
from xtuner.v1.loss import CELossConfig
from xtuner.v1.model import Qwen3_5_VLDense27BConfig
from xtuner.v1.model.compose.qwen3_vl.modeling_qwen3_vl import (
    QWEN3VL_COMPILE_CFG,
)
from xtuner.v1.module.rope import RopeParametersConfig
from xtuner.v1.train import ResumeConfig, TrainerConfig


# Keep the shared Qwen3.5/Qwen3.8 vision-layer path eager until that compile
# path has been validated on the target PyTorch/CUDA stack. Projector and text
# model compilation remain controlled by TORCH_COMPILE.
QWEN3VL_COMPILE_CFG.pop(
    "xtuner.v1.model.compose.qwen3_vl.modeling_vision.Qwen3VLVisionLayer.forward",
    None,
)


def _get_int_env(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _get_float_env(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _get_bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ceph_config = os.getenv("CEPH_CONFIG", "")
meta_data_path = Path(os.environ["META_DATA_PATH"])
model_path = Path(os.environ["MODEL_PATH"])
work_dir = Path(os.environ["WORK_DIR"])
tokenizer_cache_dir = os.environ["TOKENIZER_CACHE_DIR"]

if not meta_data_path.is_file():
    raise FileNotFoundError(f"META_DATA_PATH does not exist: {meta_data_path}")
if not (model_path / "config.json").is_file():
    raise FileNotFoundError(f"MODEL_PATH does not contain config.json: {model_path}")
if not (model_path / "model.safetensors.index.json").is_file():
    raise FileNotFoundError(f"MODEL_PATH does not contain model.safetensors.index.json: {model_path}")

work_dir.mkdir(parents=True, exist_ok=True)
shutil.copy2(__file__, work_dir)


# ---------------------------------------------------------------------------
# Qwen3.8 chat template
# ---------------------------------------------------------------------------

chat_template_name = os.getenv("CHAT_TEMPLATE_NAME", "qwen3.8-vl")
enable_thinking = _get_bool_env("ENABLE_THINKING", True)
reasoning_effort = os.getenv("REASONING_EFFORT", "xhigh")
preserve_thinking = _get_bool_env("PRESERVE_THINKING", True)

if chat_template_name != "qwen3.8-vl":
    raise ValueError(f"Qwen3.8-27B must use CHAT_TEMPLATE_NAME=qwen3.8-vl, got {chat_template_name!r}")
if reasoning_effort not in {"xhigh", "medium", "low"}:
    raise ValueError(f"REASONING_EFFORT must be one of xhigh, medium, or low, got {reasoning_effort!r}")


# ---------------------------------------------------------------------------
# Sequence, packing, optimization, and parallelism
# ---------------------------------------------------------------------------

sample_max_length = _get_int_env("SAMPLE_MAX_LENGTH", 256 * 1024)
pack_max_length = _get_int_env("PACK_MAX_LENGTH", 256 * 1024)
native_context_length = 262_144
max_extended_context_length = 1_000_000
max_position_embeddings = _get_int_env(
    "MAX_POSITION_EMBEDDINGS",
    max(native_context_length, sample_max_length, pack_max_length),
)

if sample_max_length < 1 or pack_max_length < 1:
    raise ValueError("SAMPLE_MAX_LENGTH and PACK_MAX_LENGTH must be positive")
if max_position_embeddings > max_extended_context_length:
    raise ValueError(
        f"MAX_POSITION_EMBEDDINGS={max_position_embeddings} exceeds Qwen3.8's documented extended limit "
        f"{max_extended_context_length}"
    )
if sample_max_length > max_position_embeddings:
    raise ValueError(
        f"SAMPLE_MAX_LENGTH={sample_max_length} exceeds MAX_POSITION_EMBEDDINGS={max_position_embeddings}"
    )
if pack_max_length > max_position_embeddings:
    raise ValueError(f"PACK_MAX_LENGTH={pack_max_length} exceeds MAX_POSITION_EMBEDDINGS={max_position_embeddings}")

use_yarn = max_position_embeddings > native_context_length
rope_scaling_factor = _get_float_env("ROPE_SCALING_FACTOR", 4.0)
if use_yarn and rope_scaling_factor <= 1.0:
    raise ValueError(f"ROPE_SCALING_FACTOR must be greater than 1 for extended context, got {rope_scaling_factor}")
if use_yarn and max_position_embeddings > int(native_context_length * rope_scaling_factor):
    raise ValueError(
        f"ROPE_SCALING_FACTOR={rope_scaling_factor} only covers "
        f"{int(native_context_length * rope_scaling_factor)} positions, below "
        f"MAX_POSITION_EMBEDDINGS={max_position_embeddings}"
    )

rand_video_max_frames = _get_int_env("RAND_VIDEO_MAX_FRAMES", 24)
num_workers = _get_int_env("NUM_WORKERS", 4)
global_batch_size = _get_int_env("GLOBAL_BATCH_SIZE", 8)

# TOTAL_STEP is intended for short end-to-end smoke runs. When it is set, it
# takes the place of TOTAL_EPOCH because TrainerConfig requires exactly one of
# the two stopping conditions. Normal training remains epoch-based by default.
total_step_value = os.getenv("TOTAL_STEP")
if total_step_value in {None, ""}:
    total_step = None
    total_epoch = _get_int_env("TOTAL_EPOCH", 1)
else:
    total_step = int(total_step_value)
    if total_step < 1:
        raise ValueError(f"TOTAL_STEP must be positive, got {total_step}")
    total_epoch = None

hf_interval = _get_int_env("HF_INTERVAL", 500)
hf_max_keep = _get_int_env("HF_MAX_KEEP", 2)
checkpoint_interval = _get_int_env("CHECKPOINT_INTERVAL", 500)
checkpoint_maxkeep = _get_int_env("CHECKPOINT_MAXKEEP", 2)

lr = _get_float_env("LR", 2e-5)
lr_min = _get_float_env("LR_MIN", 1e-6)
weight_decay = _get_float_env("WEIGHT_DECAY", 0.05)
warmup_ratio = _get_float_env("WARMUP_RATIO", 0.1)
recompute_ratio = _get_float_env("RECOMPUTE_RATIO", 1.0)
vision_recompute_ratio = _get_float_env("VISION_RECOMPUTE_RATIO", 1.0)
loss_reduction = os.getenv("LOSS_REDUCTION", "square")
max_pixels = _get_int_env("MAX_PIXELS", 16_777_216)

# On 32 GPUs, SP=4 and TP=1 produce DP=8. GLOBAL_BATCH_SIZE=8 therefore
# gives one packed sample to each data-parallel replica per global step.
sp_size = _get_int_env("SP_SIZE", 4)
tp_size = _get_int_env("TP_SIZE", 1)
torch_compile = _get_bool_env("TORCH_COMPILE", True)
enable_fp8 = _get_bool_env("ENABLE_FP8", False)

if sp_size < 1:
    raise ValueError(f"SP_SIZE must be positive, got {sp_size}")
if tp_size != 1:
    raise ValueError(f"Qwen3.8-27B dense tensor parallelism is not implemented; TP_SIZE must be 1, got {tp_size}")
if sp_size not in {1, 2, 4, 8}:
    raise ValueError(
        "Qwen3.8-27B currently supports SP_SIZE in {1, 2, 4, 8}. "
        "The limit comes from the common head/channel divisors of its gated attention and GatedDeltaNet layers."
    )

# Qwen3.8-27B is dense, so expert parallelism must remain disabled.
ep_size = 1


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

with (model_path / "config.json").open("r", encoding="utf-8") as file:
    model_hf_config: dict[str, Any] = json.load(file)

if model_hf_config.get("model_type") != "qwen3_5":
    raise ValueError(f"Expected MODEL_PATH config model_type='qwen3_5', got {model_hf_config.get('model_type')!r}")

architectures = model_hf_config.get("architectures", [])
if "Qwen3_5ForConditionalGeneration" not in architectures:
    raise ValueError(f"MODEL_PATH is not a Qwen3.8 conditional-generation checkpoint: architectures={architectures!r}")

text_hf_config = model_hf_config["text_config"]
vision_hf_config = model_hf_config["vision_config"]

expected_text_shape = {
    "vocab_size": 248320,
    "num_hidden_layers": 64,
    "hidden_size": 5120,
    "intermediate_size": 17408,
    "num_attention_heads": 24,
    "num_key_value_heads": 4,
    "head_dim": 256,
}
actual_text_shape = {key: text_hf_config.get(key) for key in expected_text_shape}
if actual_text_shape != expected_text_shape:
    raise ValueError(
        f"MODEL_PATH does not match Qwen3.8-27B text config. Expected {expected_text_shape}, got {actual_text_shape}"
    )

expected_vision_shape = {
    "depth": 27,
    "hidden_size": 1152,
    "intermediate_size": 4304,
    "out_hidden_size": 5120,
}
actual_vision_shape = {key: vision_hf_config.get(key) for key in expected_vision_shape}
if actual_vision_shape != expected_vision_shape:
    raise ValueError(
        "MODEL_PATH does not match Qwen3.8-27B vision config. "
        f"Expected {expected_vision_shape}, got {actual_vision_shape}"
    )

model_cfg = Qwen3_5_VLDense27BConfig(
    only_llm_forward=_get_bool_env("ONLY_LLM_FORWARD", False),
    compile_cfg=torch_compile,
)
model_cfg.text_config.vocab_size = text_hf_config["vocab_size"]
model_cfg.text_config.max_position_embeddings = max_position_embeddings

if use_yarn:
    # Official Qwen3.8 1M extension: retain interleaved mRoPE while applying
    # static YaRN to the rotary dimensions.
    model_cfg.text_config.rope_parameters_cfg = RopeParametersConfig(
        rope_theta=10_000_000.0,
        rope_type="yarn",
        mrope_section=[11, 11, 10],
        partial_rotary_factor=0.25,
        factor=rope_scaling_factor,
        original_max_position_embeddings=native_context_length,
    )

if enable_fp8:
    model_cfg.text_config.float8_cfg = Float8Config(
        scaling_granularity_gemm=ScalingGranularity.TILEWISE,
        scaling_granularity_grouped_gemm=ScalingGranularity.TILEWISE,
    )

# Do not configure MTP here. The official checkpoint's 15 mtp.* tensors are
# deliberately skipped while dense MTP remains unimplemented.


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

oss_loader_cfg = OSSLoaderConfig(backend_kwargs={"conf_path": ceph_config}) if ceph_config else None

ds_collections: dict[str, Any] = json.loads(meta_data_path.read_text(encoding="utf-8"))
has_pretrain = any(data.get("text_pretrain", False) for data in ds_collections.values())

# Keep this distinct from every Qwen3.5 cache tag because Qwen3.8 rendering and
# loss-mask semantics are intentionally different.
tokenize_cache_tag = (
    f"qwen38_dense27b_v1_sample{sample_max_length}_ctx{max_position_embeddings}_"
    f"yarn{rope_scaling_factor if use_yarn else 0}_"
    f"{reasoning_effort}_thinking{int(enable_thinking)}_preserve{int(preserve_thinking)}"
)

dataset_config: list[dict[str, Any]] = []

for name, data in ds_collections.items():
    is_pretrain = data.get("text_pretrain", False)

    if is_pretrain:
        tokenize_fn = PretrainTokenizeFunctionConfig(hash=data.get("hash"))
    else:
        tokenize_fn = Qwen3VLTokenizeFnConfig(
            chat_template=chat_template_name,
            llm_pack_weight=-3.2,
            visual_pack_weight=5.0,
            max_length=sample_max_length,
            processor_path=str(model_path),
            rand_video_max_frames=rand_video_max_frames,
            oss_loader_cfg=oss_loader_cfg,
            max_pixels=max_pixels,
            add_generation_prompt=False,
            add_vision_id=True,
            enable_thinking=enable_thinking,
            reasoning_effort=reasoning_effort,
            preserve_thinking=preserve_thinking,
            debug=_get_bool_env("TOKENIZE_DEBUG", True),
        )

    dataset_config.append(
        {
            "dataset": DatasetConfig(
                name=name,
                anno_path=data["annotation"],
                media_root=data.get("media_root") or "",
                sample_ratio=data.get("sample_ratio", 1.0),
                class_name=("JsonlDataset" if is_pretrain else "VLMJsonlDataset"),
                enable_sequential_sampler=True,
                cache_tag=tokenize_cache_tag,
                cache_dir=tokenizer_cache_dir,
            ),
            "tokenize_fn": tokenize_fn,
        }
    )

dataloader_config = DataloaderConfig(
    dataset_config_list=dataset_config,
    pack_max_length=pack_max_length,
    pack_level="mllm_hybrid" if has_pretrain else "soft",
    pack_to_max_length=True,
    collator="qwen3_vl_sft_collator",
    num_workers=num_workers,
    pack_extra_buffer_size=_get_int_env("PACK_EXTRA_BUFFER_SIZE", 20),
)


# ---------------------------------------------------------------------------
# Optimizer, LR, FSDP, loss, and trainer
# ---------------------------------------------------------------------------

optim_cfg = MuonConfig(
    lr=lr,
    weight_decay=weight_decay,
)

lr_cfg = LRConfig(
    lr_type="cosine",
    warmup_ratio=warmup_ratio,
    lr_min=lr_min,
)

fsdp_cfg = FSDPConfig(
    tp_size=tp_size,
    ep_size=ep_size,
    recompute_ratio=recompute_ratio,
    vision_recompute_ratio=vision_recompute_ratio,
    torch_compile=torch_compile,
    checkpoint_preserve_rng_state=False,
)

loss_cfg = CELossConfig(
    mode="chunk",
    chunk_size=_get_int_env("LOSS_CHUNK_SIZE", 1024),
    loss_reduction=loss_reduction,
)

trainer = TrainerConfig(
    sp_size=sp_size,
    load_from=str(model_path),
    resume_cfg=ResumeConfig(auto_resume=True),
    tokenizer_path=str(model_path),
    fsdp_cfg=fsdp_cfg,
    exp_tracker="tensorboard",
    model_cfg=model_cfg,
    optim_cfg=optim_cfg,
    dataloader_cfg=dataloader_config,
    lr_cfg=lr_cfg,
    loss_cfg=loss_cfg,
    global_batch_size=global_batch_size,
    total_step=total_step,
    total_epoch=total_epoch,
    hf_interval=hf_interval,
    checkpoint_interval=checkpoint_interval,
    checkpoint_maxkeep=checkpoint_maxkeep,
    hf_max_keep=hf_max_keep,
    work_dir=work_dir,
    debug_skip_save=_get_bool_env("DEBUG_SKIP_SAVE", False),
)
