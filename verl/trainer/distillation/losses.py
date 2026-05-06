# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch
from tensordict import TensorDict

from verl.base_config import BaseConfig
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.metric import AggregationType, Metric
from verl.workers.config import ActorConfig, DistillationConfig, DistillationLossConfig
from verl.workers.utils.losses import ppo_loss
from verl.workers.utils.padding import no_padding_2_padding

DistillationLossFn = Callable[
    [
        ActorConfig,  # actor_config
        DistillationConfig,  # distillation_config
        dict,  # model_output
        TensorDict,  # micro batch input
    ],
    tuple[torch.Tensor, dict[str, Any]],
]


def is_distillation_enabled(config: Optional[DistillationConfig]) -> bool:
    """Check if distillation is enabled based on the provided configuration."""
    if config is None:
        return False
    return config.enabled


@dataclass
class DistillationLossSettings(BaseConfig):
    """
    Settings for a distillation loss function to be registered.

    Args:
        names (str | list[str]): Name(s) to register the distillation loss function under.
        use_topk (bool): Whether the loss function uses top-k log probabilities.
        use_estimator (bool): Whether the loss function uses single-sample KL estimators.
    """

    names: str | list[str] = field(default_factory=list)
    use_topk: bool = False
    use_estimator: bool = False

    _mutable_fields = {"names"}

    def __post_init__(self):
        self.names = [self.names] if isinstance(self.names, str) else self.names
        if sum([self.use_topk, self.use_estimator]) != 1:
            raise ValueError(
                f"Expected only one of use_estimator, use_topk, but got {self.use_estimator=}, {self.use_topk=}."
            )


DISTILLATION_LOSS_REGISTRY: dict[str, DistillationLossFn] = {}
DISTILLATION_SETTINGS_REGISTRY: dict[str, DistillationLossSettings] = {}


def register_distillation_loss(
    loss_settings: DistillationLossSettings,
) -> Callable[[DistillationLossFn], DistillationLossFn]:
    """Register a distillation loss function with the given name."""

    def decorator(func: DistillationLossFn) -> DistillationLossFn:
        for name in loss_settings.names:
            if name in DISTILLATION_LOSS_REGISTRY:
                raise ValueError(f"Distillation loss function with name '{name}' is already registered.")
            DISTILLATION_LOSS_REGISTRY[name] = func
            DISTILLATION_SETTINGS_REGISTRY[name] = loss_settings
        return func

    return decorator


def get_distillation_loss_fn(loss_name: str) -> DistillationLossFn:
    """Get the distillation loss function with a given name."""
    if loss_name not in DISTILLATION_LOSS_REGISTRY:
        raise ValueError(
            f"Unsupported loss mode: {loss_name}. Supported modes are: {list(DISTILLATION_LOSS_REGISTRY.keys())}"
        )
    return DISTILLATION_LOSS_REGISTRY[loss_name]


def get_distillation_loss_settings(loss_name: str) -> DistillationLossSettings:
    """Get the distillation loss settings with a given name."""
    if loss_name not in DISTILLATION_SETTINGS_REGISTRY:
        raise ValueError(
            f"Unsupported loss mode: {loss_name}. Supported modes are: {list(DISTILLATION_SETTINGS_REGISTRY.keys())}"
        )
    return DISTILLATION_SETTINGS_REGISTRY[loss_name]


def compute_distillation_loss_range(
    distillation_losses: torch.Tensor, response_mask: torch.Tensor
) -> dict[str, Metric]:
    """Compute min and max distillation loss over valid response tokens."""
    distillation_losses_response = distillation_losses[response_mask.bool()]
    return {
        "distillation/loss_min": Metric(AggregationType.MIN, distillation_losses_response.min()),
        "distillation/loss_max": Metric(AggregationType.MAX, distillation_losses_response.max()),
    }


def compute_topk_loss(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    data: TensorDict,
    student_logits: torch.Tensor,
    data_format: str,
) -> torch.Tensor:
    """Compute the topk loss in logit processor.

    Returns:
    - distillation_losses: (bsz, seqlen/cp_size)
    - student_mass: (bsz, seqlen/cp_size)
    - teacher_mass: (bsz, seqlen/cp_size)
    """
    match config.strategy:
        case "fsdp":
            import verl.trainer.distillation.fsdp.losses as fsdp_losses

            distillation_loss_fn = fsdp_losses.compute_forward_kl_topk
        case "megatron":
            import verl.trainer.distillation.megatron.losses as megatron_losses

            distillation_loss_fn = megatron_losses.compute_forward_kl_topk
        case _:
            raise NotImplementedError(f"Unsupported strategy: {config.strategy=}")

    outputs = distillation_loss_fn(
        student_logits=student_logits,
        teacher_topk_log_probs=data["teacher_logprobs"],
        teacher_topk_ids=data["teacher_ids"],
        config=distillation_config,
        data_format=data_format,
    )

    expected_shape = student_logits.shape[:2]
    for k, v in outputs.items():
        assert v.shape == expected_shape, f"Expected shape {expected_shape}, but got {v.shape} for {k=}."

    return outputs


def distillation_ppo_loss(
    config: ActorConfig,
    distillation_config: Optional[DistillationConfig],
    model_output: dict = None,
    data: TensorDict = None,
    dp_group=None,
    student_logits: torch.Tensor = None,
    data_format: str = "thd",
):
    """Loss function used both for logit processor and final policy loss.
    - student_logits is not None, compute the topk loss in logit processor.
    - student_logits is None, compute final policy loss.

    [split sequence across sp/cp groups]
                   |
    [model forward and output logits: (bsz, seqlen/cp_size, vocab_size/tp_size)]
                   |
    [logits processor compute topk loss: (bsz, seqlen/cp_size)]
                   |
    [all gather topk loss across sp/cp groups: (bsz, seqlen)]
                   |
    [combine topk loss with policy loss]

    Args:
        config: Actor configuration.
        distillation_config: Distillation configuration.
        model_output: Model output, including log_probs, entropy.
        data: Micro input batch, contains
          - teacher_logprobs: (bsz, seqlen, topk)
          - teacher_ids: (bsz, seqlen, topk)
        student_logits: (bsz, seqlen/cp_size, vocab_size/tp_size).
        data_format: "thd" or "bshd", models not support THD format, e.g GPT-OSS, Qwen3.5

    Returns:
    - student_logits is not None, return the topk loss tensor (bsz, seqlen/cp_size).
    - student_logits is None, return the final policy loss scalar and metrics.
    """

    # Called as logits processor
    if student_logits is not None:
        return compute_topk_loss(config, distillation_config, data, student_logits, data_format)

    # Called as final policy loss
    distillation_loss_config = distillation_config.distillation_loss

    # RLAD mode: unified loss that replaces both policy + distillation
    if distillation_loss_config.loss_mode == "rlad":
        rlad_loss_fn = get_distillation_loss_fn("rlad")
        rlad_loss, rlad_metrics = rlad_loss_fn(
            config=config,
            distillation_config=distillation_config,
            model_output=model_output,
            data=data,
        )
        # RLAD returns the full unified loss — no separate policy loss needed
        return rlad_loss, rlad_metrics

    distill_loss, distill_metrics = distillation_loss(config, distillation_config, model_output, data)
    policy_loss, policy_metrics = ppo_loss(config, model_output, data, dp_group)
    if not distillation_loss_config.use_task_rewards:
        policy_loss = 0.0

    # Combine distillation with policy loss
    policy_metrics.update(distill_metrics)
    distillation_loss_coef = (
        distillation_loss_config.distillation_loss_coef if distillation_loss_config.use_task_rewards else 1.0
    )

    # Adaptive distill coef decay based on kl_abs_mean
    import os as _os
    if _os.environ.get("ADAPTIVE_DISTILL_COEF", "") == "1":
        _kl_off = float(_os.environ.get("ADAPTIVE_DISTILL_KL_OFF", "0.7"))
        _kl_decay_start = float(_os.environ.get("ADAPTIVE_DISTILL_KL_DECAY_START", "0.3"))
        _kl_metric = distill_metrics.get("bt_opd/kl_abs_mean")
        if _kl_metric is not None and len(_kl_metric.values) > 0:
            import torch as _torch
            _raw = _kl_metric.values[0]
            _kl_val = _raw.item() if isinstance(_raw, _torch.Tensor) else float(_raw)
            _coef_floor = float(_os.environ.get("ADAPTIVE_DISTILL_COEF_FLOOR", "0.0"))
            if _kl_val >= _kl_off:
                # kl too high — reduce to floor (0.0 = fully off, >0 = maintain minimum connection)
                distillation_loss_coef = _coef_floor
                print(f"[Adaptive-Distill] OFF→floor={_coef_floor:.2f}: kl_abs_mean={_kl_val:.4f} >= {_kl_off}")
            elif _kl_val > _kl_decay_start:
                # Linear decay: coef * (kl_off - kl) / (kl_off - kl_decay_start)
                _scale = (_kl_off - _kl_val) / (_kl_off - _kl_decay_start)
                distillation_loss_coef = distillation_loss_coef * _scale
                print(f"[Adaptive-Distill] DECAY: kl={_kl_val:.4f}, scale={_scale:.3f}, coef={distillation_loss_coef:.3f}")
            policy_metrics["distillation/adaptive_coef"] = Metric(AggregationType.MEAN, distillation_loss_coef)

    policy_loss += distill_loss * distillation_loss_coef
    policy_metrics["distillation/loss"] = Metric(value=distill_loss, aggregation=AggregationType.SUM)

    return policy_loss, policy_metrics


def distillation_loss(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output: dict,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the distillation loss and related metrics.

    Returns:
    - distillation_loss: Aggregated distillation loss scalar.
    - distillation_metrics: Dictionary of metrics.
    """
    assert distillation_config is not None
    loss_config: DistillationLossConfig = distillation_config.distillation_loss
    distillation_loss_fn = get_distillation_loss_fn(loss_config.loss_mode)
    distillation_losses, distillation_metrics = distillation_loss_fn(
        config=config,
        distillation_config=distillation_config,
        model_output=model_output,
        data=data,
    )
    response_mask = data["response_mask"]
    loss_agg_mode = config.loss_agg_mode

    distillation_metrics.update(
        compute_distillation_loss_range(distillation_losses=distillation_losses, response_mask=response_mask)
    )
    if loss_config.loss_max_clamp is not None:
        # clamping min is for k1 loss which can be negative
        distillation_losses = distillation_losses.clamp(min=-loss_config.loss_max_clamp, max=loss_config.loss_max_clamp)

    if loss_config.use_policy_gradient:
        # Use negative distillation loss as reward, as done by https://thinkingmachines.ai/blog/on-policy-distillation/.
        policy_loss_fn = get_policy_loss_fn(loss_config.policy_loss_mode)
        for k, v in config.global_batch_info.items():
            loss_config.global_batch_info[k] = v
        log_prob = no_padding_2_padding(model_output["log_probs"], data)
        old_log_prob = data["old_log_probs"]
        rollout_is_weights = data.get("rollout_is_weights", None)
        distillation_loss, pg_metrics = policy_loss_fn(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=-distillation_losses.detach(),
            response_mask=response_mask,
            loss_agg_mode=loss_agg_mode,
            config=loss_config,
            rollout_is_weights=rollout_is_weights,
        )
        pg_metrics = {f"distillation/{k[len('actor/') :]}": v for k, v in pg_metrics.items()}
        distillation_metrics.update(pg_metrics)
    else:
        # Directly backpropagate distillation loss as a supervised loss, as in https://arxiv.org/abs/2306.13649.
        distillation_loss = agg_loss(
            loss_mat=distillation_losses,
            loss_mask=response_mask,
            loss_agg_mode=loss_agg_mode,
            **config.global_batch_info,
        )

    return distillation_loss, distillation_metrics


@register_distillation_loss(DistillationLossSettings(names=["forward_kl_topk"], use_topk=True))  # type: ignore[arg-type]
def compute_forward_kl_topk(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output: dict,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute forward KL distillation loss and related metrics using top-k log probabilities.

    Returns:
    - distillation_losses: (bsz, resp_len)
    - distillation_metrics: Dictionary of metrics.
    """
    # topk loss has been computed in logits processor
    distillation_losses = no_padding_2_padding(model_output["distillation_losses"], data)
    student_mass = no_padding_2_padding(model_output["student_mass"], data)
    teacher_mass = no_padding_2_padding(model_output["teacher_mass"], data)
    response_mask_bool = data["response_mask"].bool()
    assert distillation_losses.shape == student_mass.shape == teacher_mass.shape == response_mask_bool.shape

    # Log amount of mass in the top-k log probabilities for both student and teacher.
    student_mass = student_mass[response_mask_bool]
    teacher_mass = teacher_mass[response_mask_bool]
    distillation_metrics = {
        "distillation/student_mass": student_mass.mean().item(),
        "distillation/student_mass_min": Metric(AggregationType.MIN, student_mass.min()),
        "distillation/student_mass_max": Metric(AggregationType.MAX, student_mass.max()),
        "distillation/teacher_mass": teacher_mass.mean().item(),
        "distillation/teacher_mass_min": Metric(AggregationType.MIN, teacher_mass.min()),
        "distillation/teacher_mass_max": Metric(AggregationType.MAX, teacher_mass.max()),
    }

    # Due to use of top-k, student and teacher distributions don't sum to 1 -> divergences can be negative.
    distillation_losses = distillation_losses.clamp_min(0.0)

    return distillation_losses, distillation_metrics


@register_distillation_loss(
    DistillationLossSettings(names=["kl", "k1", "abs", "mse", "k2", "low_var_kl", "k3"], use_estimator=True)
)  # type: ignore[arg-type]
def compute_distillation_loss_reverse_kl_estimator(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the distillation loss and related metrics using single-sample KL estimators.

    Uses the kl_penalty function from core_algos which supports various KL divergence
    estimators: "kl", "k1", "abs", "mse", "k2", "low_var_kl", "k3".

    Returns:
    - distillation_losses: (bsz, resp_len)
    - distillation_metrics: Dictionary of metrics.
    """
    student_log_probs = no_padding_2_padding(model_output["log_probs"], data)
    teacher_log_probs = no_padding_2_padding(data["teacher_logprobs"], data).squeeze(-1)
    response_mask_bool = data["response_mask"].bool()
    assert teacher_log_probs.shape == student_log_probs.shape == response_mask_bool.shape

    loss_config: DistillationLossConfig = distillation_config.distillation_loss
    distillation_losses = kl_penalty(
        logprob=student_log_probs, ref_logprob=teacher_log_probs, kl_penalty=loss_config.loss_mode
    )
    # Since k1 can be negative, log the mean absolute loss.
    metrics = {
        "distillation/abs_loss": Metric(AggregationType.MEAN, distillation_losses[response_mask_bool].abs().mean()),
    }
    return distillation_losses, metrics


# ── BT-OPD: Bidirectional Truncated On-Policy Distillation ──────────
# Turn-aware truncation + bidirectional corrective gradient.
# Config via env vars: BT_OPD_MAX_TURN (default 3), BT_OPD_BIDIRECTIONAL (default 1).

_BT_OPD_MAX_TURN = int(os.environ.get("BT_OPD_MAX_TURN", "3"))
_BT_OPD_BIDIRECTIONAL = os.environ.get("BT_OPD_BIDIRECTIONAL", "1") == "1"

# Top-K position filtering (EMA-PG inspired, arXiv:2602.04417):
# Only distill on positions where teacher is confident (high logprob).
# Filters out tail positions where student-teacher gap is large → prevents divergence.
# BT_OPD_TOPK_RATIO=1.0 (default) = no filtering; 0.5 = keep top 50% positions by teacher confidence.
_BT_OPD_TOPK_RATIO = float(os.environ.get("BT_OPD_TOPK_RATIO", "1.0"))

# Resolve <|im_start|> token ID dynamically from tokenizer to avoid hardcoding
# model-specific values. Falls back to env var BT_OPD_IM_START_TOKEN_ID if set.
_IM_START_TOKEN_ID: int | None = None


def _get_im_start_token_id() -> int:
    """Lazily resolve <|im_start|> token ID from the model's tokenizer."""
    global _IM_START_TOKEN_ID
    if _IM_START_TOKEN_ID is not None:
        return _IM_START_TOKEN_ID

    # Allow env var override for cases where tokenizer isn't available
    env_val = os.environ.get("BT_OPD_IM_START_TOKEN_ID")
    if env_val is not None:
        _IM_START_TOKEN_ID = int(env_val)
        return _IM_START_TOKEN_ID

    # Try to resolve from the model tokenizer
    model_path = os.environ.get("BT_OPD_MODEL_PATH", "")
    if model_path:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            _IM_START_TOKEN_ID = tokenizer.convert_tokens_to_ids("<|im_start|>")
            print(f"[BT-OPD] Resolved <|im_start|> token ID = {_IM_START_TOKEN_ID} from {model_path}")
            return _IM_START_TOKEN_ID
        except Exception as e:
            print(f"[BT-OPD] Warning: Failed to resolve <|im_start|> from tokenizer: {e}")

    # Fallback: Qwen3.5 default
    _IM_START_TOKEN_ID = 248045
    print(f"[BT-OPD] Using default <|im_start|> token ID = {_IM_START_TOKEN_ID} (Qwen3.5)")
    return _IM_START_TOKEN_ID


def _build_turn_mask(input_ids, response_mask_bool, max_turn):
    """Build per-token mask that is 1.0 for first `max_turn` assistant turns."""
    if max_turn <= 0:
        return torch.ones_like(response_mask_bool, dtype=torch.float32)

    resp_indices = response_mask_bool.nonzero(as_tuple=True)[0]
    if len(resp_indices) == 0:
        return torch.zeros_like(response_mask_bool, dtype=torch.float32)

    resp_start = resp_indices[0].item()
    resp_end = resp_indices[-1].item() + 1

    resp_ids = input_ids[resp_start:resp_end]
    im_starts = (resp_ids == _get_im_start_token_id()).nonzero(as_tuple=True)[0]

    mask = torch.zeros_like(response_mask_bool, dtype=torch.float32)

    if len(im_starts) == 0:
        if max_turn >= 1:
            mask[resp_start:resp_end] = 1.0
        return mask

    # Turn 0: resp_start → first <|im_start|>
    boundaries = [(resp_start, resp_start + im_starts[0].item())]
    for i in range(len(im_starts)):
        s = resp_start + im_starts[i].item()
        e = (resp_start + im_starts[i + 1].item()) if i + 1 < len(im_starts) else resp_end
        boundaries.append((s, e))

    for tidx, (s, e) in enumerate(boundaries):
        if tidx >= max_turn:
            break
        mask[s:e] = 1.0

    return mask


@register_distillation_loss(
    DistillationLossSettings(names=["bt_opd_kl"], use_estimator=True)  # type: ignore[arg-type]
)
def compute_bt_opd_loss(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output: dict,
    data: TensorDict,
) -> tuple[torch.Tensor, dict]:
    """BT-OPD: Bidirectional Truncated On-Policy Distillation loss.

    Extensions over standard reverse-KL:
      1. Truncated: OPD only on first K assistant turns (env BT_OPD_MAX_TURN).
      2. Bidirectional: flip KL sign for negative-reward trajectories (env BT_OPD_BIDIRECTIONAL).
    """
    student_log_probs = no_padding_2_padding(model_output["log_probs"], data)
    teacher_log_probs = no_padding_2_padding(data["teacher_logprobs"], data).squeeze(-1)
    response_mask = data["response_mask"]
    response_mask_bool = response_mask.bool()
    assert teacher_log_probs.shape == student_log_probs.shape == response_mask_bool.shape

    # Debug: check for NaN/inf in logprobs before fixing
    t_nan = teacher_log_probs.isnan().sum().item()
    t_inf = teacher_log_probs.isinf().sum().item()
    s_nan = student_log_probs.isnan().sum().item()
    s_inf = student_log_probs.isinf().sum().item()
    if t_nan > 0 or t_inf > 0 or s_nan > 0 or s_inf > 0:
        t_valid = teacher_log_probs[response_mask_bool]
        s_valid = student_log_probs[response_mask_bool]
        print(f"[BT-OPD] teacher NaN={t_nan} inf={t_inf} | student NaN={s_nan} inf={s_inf} | "
              f"teacher valid: min={t_valid.min():.4f} max={t_valid.max():.4f} nan={t_valid.isnan().sum()} | "
              f"student valid: min={s_valid.min():.4f} max={s_valid.max():.4f} nan={s_valid.isnan().sum()}")
    # Replace NaN/inf in logprobs: NaN from first-token positions, -inf from zero-probability tokens
    # Use -20.0 for NaN (≈ prob 2e-9), not 0.0 which means prob=1.0 and biases KL
    teacher_log_probs = torch.nan_to_num(teacher_log_probs, nan=-20.0, posinf=0.0, neginf=-20.0)
    student_log_probs = torch.nan_to_num(student_log_probs, nan=-20.0, posinf=0.0, neginf=-20.0)

    bsz, seq_len = student_log_probs.shape

    distillation_losses = kl_penalty(
        logprob=student_log_probs, ref_logprob=teacher_log_probs, kl_penalty="kl"
    )

    max_turn = _BT_OPD_MAX_TURN
    bidirectional = _BT_OPD_BIDIRECTIONAL
    num_truncated = 0
    num_negative = 0

    # 1. Truncated OPD: zero out tokens after turn K
    # Use response-only IDs (data["responses"]) not full input_ids (which is a nested tensor
    # with prompt+response and different indexing than response_mask_bool).
    response_ids_key = "responses" if "responses" in data else "input_ids"
    if max_turn > 0 and response_ids_key in data:
        for i in range(bsz):
            resp_ids_i = data[response_ids_key][i]
            # Debug: log turn detection for first sample of each batch
            if i == 0:
                valid_len = int(response_mask_bool[i].sum().item())
                valid_ids = resp_ids_i[:valid_len]
                im_count_valid = (valid_ids == _get_im_start_token_id()).sum().item()
                im_count_total = (resp_ids_i == _get_im_start_token_id()).sum().item()
                print(f"[BT-OPD Turn Debug] valid_im_start={im_count_valid} total_im_start={im_count_total} "
                      f"valid_len={valid_len} shape={resp_ids_i.shape}")
            tmask = _build_turn_mask(resp_ids_i, response_mask_bool[i], max_turn)
            before = response_mask_bool[i].sum().item()
            after = (response_mask_bool[i] & tmask.bool()).sum().item()
            num_truncated += before - after
            distillation_losses[i] = distillation_losses[i] * tmask.to(distillation_losses.device)

    # 2. Bidirectional OPD: flip KL for negative-advantage trajectories
    if bidirectional and "advantages" in data:
        for i in range(bsz):
            valid_adv = data["advantages"][i][response_mask_bool[i]]
            if len(valid_adv) > 0 and valid_adv[0].item() < 0:
                distillation_losses[i] = -distillation_losses[i]
                num_negative += 1

    # 3. Top-K position filtering: only distill on positions where teacher is confident
    topk_ratio = _BT_OPD_TOPK_RATIO
    num_topk_filtered = 0
    if topk_ratio < 1.0:
        for i in range(bsz):
            valid_mask = response_mask_bool[i]
            valid_positions = valid_mask.nonzero(as_tuple=True)[0]
            n_valid = len(valid_positions)
            if n_valid == 0:
                continue
            # Get teacher confidence at each valid position
            teacher_conf = teacher_log_probs[i, valid_positions]
            # Keep top-k% positions by teacher log prob (higher = more confident)
            k = max(1, int(n_valid * topk_ratio))
            _, topk_indices = torch.topk(teacher_conf, k, largest=True)
            # Build mask: zero out filtered positions
            topk_mask = torch.zeros(n_valid, device=distillation_losses.device)
            topk_mask[topk_indices] = 1.0
            # Apply: zero out distillation loss at low-confidence positions
            full_topk_mask = torch.zeros_like(distillation_losses[i])
            full_topk_mask[valid_positions] = topk_mask
            distillation_losses[i] = distillation_losses[i] * full_topk_mask
            num_topk_filtered += n_valid - k

    valid = distillation_losses[response_mask_bool]
    metrics = {
        "bt_opd/kl_abs_mean": Metric(AggregationType.MEAN, valid.abs().mean()),
        "bt_opd/max_turn": float(max_turn),
        "bt_opd/bidirectional": float(bidirectional),
        "bt_opd/negative_trajs": float(num_negative),
        "bt_opd/truncated_tokens": float(num_truncated),
        "bt_opd/topk_ratio": topk_ratio,
        "bt_opd/topk_filtered": float(num_topk_filtered),
    }

    return distillation_losses, metrics


# ── RLAD: Reinforcement-Aware Knowledge Distillation ────────────────
# arXiv:2602.22495 — Trust Region Ratio Distillation (TRRD)
# Unifies policy gradient and distillation into a single importance ratio:
#   log r_TRRD = α·(log π_s - log π_s_old) + (1-α)·(log π_s - log π_T)
# No separate distillation loss or distill_coef needed.
_RLAD_ALPHA = float(os.environ.get("RLAD_ALPHA", "0.5"))


@register_distillation_loss(
    DistillationLossSettings(names=["rlad"], use_estimator=True)  # type: ignore[arg-type]
)
def compute_rlad_loss(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output: dict,
    data: TensorDict,
) -> tuple[torch.Tensor, dict]:
    """RLAD: Unified TRRD ratio that replaces separate policy + distillation losses.

    Instead of: policy_loss + distill_coef * distill_loss
    RLAD uses:  clipped_surrogate_loss(r_TRRD, advantages)

    where r_TRRD = exp(α·log(π_s/π_s_old) + (1-α)·log(π_s/π_T))

    The advantage signal automatically gates teacher influence:
    - Positive advantage: teacher reinforces good actions
    - Negative advantage: teacher protects against aggressive unlearning
    - Zero advantage: teacher has no influence

    Returns the full unified loss (not just distillation part).
    distillation_ppo_loss() must skip separate policy_loss when using this mode.
    """
    import verl.utils.torch_functional as verl_F

    alpha = _RLAD_ALPHA

    # Get logprobs
    student_log_probs = no_padding_2_padding(model_output["log_probs"], data)
    teacher_log_probs = no_padding_2_padding(data["teacher_logprobs"], data).squeeze(-1)
    old_log_probs = data["old_log_probs"]
    response_mask = data["response_mask"]
    response_mask_bool = response_mask.bool()
    advantages = data["advantages"]

    assert teacher_log_probs.shape == student_log_probs.shape == response_mask_bool.shape

    # Sanitize teacher logprobs (NaN from first-token, -inf from zero-prob tokens)
    t_nan = teacher_log_probs.isnan().sum().item()
    t_inf = teacher_log_probs.isinf().sum().item()
    if t_nan > 0 or t_inf > 0:
        print(f"[RLAD] teacher NaN={t_nan} inf={t_inf}")
    teacher_log_probs = torch.nan_to_num(teacher_log_probs, nan=-20.0, posinf=0.0, neginf=-20.0)
    student_log_probs = torch.nan_to_num(student_log_probs, nan=-20.0, posinf=0.0, neginf=-20.0)

    # TRRD ratio (Eq. 5 from paper):
    # log r_TRRD = α·(log π_s - log π_s_old) + (1-α)·(log π_s - log π_T)
    log_ratio_grpo = student_log_probs - old_log_probs    # standard GRPO ratio
    log_ratio_teacher = student_log_probs - teacher_log_probs  # student-to-teacher ratio

    log_ratio_trrd = alpha * log_ratio_grpo + (1.0 - alpha) * log_ratio_teacher

    # Clamp for numerical stability
    log_ratio_trrd = torch.clamp(log_ratio_trrd, min=-20.0, max=20.0)
    ratio_trrd = torch.exp(log_ratio_trrd)

    # PPO clipping on TRRD ratio (same as vanilla GRPO, using config clip_ratio)
    clip_ratio = config.clip_ratio
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio

    pg_losses1 = -advantages * ratio_trrd
    pg_losses2 = -advantages * torch.clamp(
        ratio_trrd, 1 - clip_ratio_low, 1 + clip_ratio_high
    )
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)

    # Dual-clip PPO for negative advantages
    clip_ratio_c = config.get("clip_ratio_c", 3.0) if hasattr(config, 'get') else 3.0
    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)

    # Aggregate
    loss_agg_mode = config.loss_agg_mode
    rlad_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode,
        **config.global_batch_info,
    )

    # Metrics
    pg_clipfrac = verl_F.masked_mean(
        torch.gt(pg_losses2, pg_losses1).float(), response_mask
    )
    ppo_kl_grpo = verl_F.masked_mean(-log_ratio_grpo, response_mask)
    ppo_kl_teacher = verl_F.masked_mean(log_ratio_teacher, response_mask)
    ppo_kl_trrd = verl_F.masked_mean(-log_ratio_trrd, response_mask)

    metrics = {
        "rlad/alpha": alpha,
        "rlad/kl_grpo": Metric(AggregationType.MEAN, ppo_kl_grpo.detach()),
        "rlad/kl_teacher": Metric(AggregationType.MEAN, ppo_kl_teacher.detach()),
        "rlad/kl_trrd": Metric(AggregationType.MEAN, ppo_kl_trrd.detach()),
        "rlad/pg_clipfrac": Metric(AggregationType.MEAN, pg_clipfrac.detach()),
        "rlad/ratio_mean": Metric(AggregationType.MEAN, verl_F.masked_mean(ratio_trrd, response_mask).detach()),
    }

    # Return rlad_loss as "distillation_losses" — but it's already the full unified loss.
    # distillation_ppo_loss() will detect rlad mode and skip separate policy loss.
    return rlad_loss, metrics
