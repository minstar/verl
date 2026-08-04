# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
import asyncio
import math
import os
from typing import Any, Optional
from uuid import uuid4

import ray
import torch
from omegaconf import DictConfig
from tensordict import TensorDict
from torch.nn import functional as F

from verl.experimental.agent_loop import AsyncLLMServerManager
from verl.protocol import DataProto
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.tokenizer import normalize_token_ids
from verl.workers.config import DistillationConfig, DistillationLossConfig

# ── Hint-based OPD configuration ──
_HINT_OPD_ENABLED = os.environ.get("HINT_OPD_ENABLED", "0") == "1"
_HINT_CORRECT = os.environ.get("HINT_OPD_CORRECT",
    "Hint: The model's reasoning and answer are correct. Reinforce this approach.")
_HINT_INCORRECT = os.environ.get("HINT_OPD_INCORRECT",
    "Hint: The model's answer is incorrect. Reconsider the reasoning and choose the correct option.")
_IM_START_TOKEN_ID: int | None = None
_IM_END_TOKEN_ID: int | None = None
# Rendered hint token ids, keyed by is_correct. The hint text is fixed per run.
_HINT_TOKEN_CACHE: dict[bool, list[int]] = {}
# A trajectory reward strictly above this counts as correct. The batch path has
# always used `score > 0`; keep that as the default so hint semantics are
# unchanged, but name it instead of burying a bare literal. Under the hcgym
# cosine reward a correct trajectory scores >= 0.7 and a wrong one scores in
# [-0.5, 0.0], so 0.0 separates them cleanly.
_HINT_OPD_CORRECT_THRESHOLD = float(os.environ.get("HINT_OPD_CORRECT_THRESHOLD", "0.0"))

# ── OPSD gold-answer conditioning (arXiv:2601.18734 baseline) ──
# Instead of the outcome-conditioned static hints above, condition the teacher on
# the sample's ground-truth answer (OPSD's privileged context). Mutually exclusive
# with hint injection: when enabled, gold conditioning takes precedence.
_OPSD_GOLD_CONDITIONING = os.environ.get("OPSD_GOLD_CONDITIONING", "0") == "1"
# The wording follows OPSD's teacher prompt (their data_collator.py transition_prompt),
# minus the problem statement, which the teacher already sees as the original prompt.
_OPSD_GOLD_TEMPLATE = os.environ.get(
    "OPSD_GOLD_TEMPLATE",
    "Here is a reference solution to the problem above:\n"
    "=== Reference Solution Begin ===\n{gold}\n=== Reference Solution End ===\n\n"
    "After reading the reference solution above, make sure you truly understand the "
    "reasoning behind each step — do not copy or paraphrase it. Now, using your own "
    "words and independent reasoning, derive the same final answer to the problem above.",
)
# Cap the injected gold text so the teacher context stays within its window.
_OPSD_GOLD_MAX_CHARS = int(os.environ.get("OPSD_GOLD_MAX_CHARS", "2000"))
# Rendered gold token ids, keyed by (truncated) gold text. Bounded: cleared if it
# ever exceeds _GOLD_TOKEN_CACHE_MAX entries (dataset-sized in practice).
_GOLD_TOKEN_CACHE: dict[str, list[int]] = {}
_GOLD_TOKEN_CACHE_MAX = 65536


def _extract_gold_text_from_fields(reward_model, extra_info) -> Optional[str]:
    """Ground truth for OPSD gold conditioning from per-sample non-tensor fields.

    Primary source: ``reward_model["ground_truth"]`` — the key the reward function
    grades against. In the hcgym data this is empty for open-ended samples, whose
    reference solution instead lives in ``extra_info["correct_answer"]``; fall back
    to it so those samples are not silently left unconditioned. Returns None when
    neither is present.
    """

    def _lookup(entry, field: str) -> Optional[str]:
        if not isinstance(entry, dict):
            return None
        value = entry.get(field)
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    return _lookup(reward_model, "ground_truth") or _lookup(extra_info, "correct_answer")


def _extract_gold_text(data: "DataProto", index: int) -> Optional[str]:
    """Per-sample ground truth for OPSD gold conditioning (batched DataProto view)."""

    def _entry(container_key: str):
        batch = data.non_tensor_batch.get(container_key)
        return None if batch is None else batch[index]

    return _extract_gold_text_from_fields(_entry("reward_model"), _entry("extra_info"))


def _render_user_turn_tokens(text: str, tokenizer) -> Optional[list[int]]:
    """Render ``text`` as a single user turn with the model's own chat template.

    Works for ChatML (Qwen), Gemma, Llama and others; falls back to literal ChatML
    only if the template cannot be applied. Any sequence-start marker the template
    prepends is dropped, because the result is spliced into the middle of an
    existing sequence.
    """
    if tokenizer is None:
        return None
    # Handle VLM processors (e.g., Qwen3VLProcessor) which wrap a text tokenizer
    text_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)

    token_ids: Optional[list[int]] = None
    try:
        rendered = text_tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=False,
        )
        bos = getattr(text_tokenizer, "bos_token", None)
        if bos and rendered.startswith(bos):
            rendered = rendered[len(bos):]
        token_ids = text_tokenizer.encode(rendered, add_special_tokens=False)
    except Exception as exc:
        print(f"[Teacher] chat template unavailable for injected turn ({exc}); using ChatML fallback")

    if not token_ids:
        content_ids = text_tokenizer.encode(f"user\n{text}", add_special_tokens=False)
        im_start, im_end = _resolve_special_token_ids(text_tokenizer)
        newline_ids = text_tokenizer.encode("\n", add_special_tokens=False)
        token_ids = [im_start] + content_ids + [im_end] + newline_ids

    return token_ids


def _stitch_out_injected_span(
    teacher_ids_full: torch.Tensor,
    teacher_logprobs_full: torch.Tensor,
    insert_pos: int,
    inject_len: int,
    target_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Remove an injected span from teacher outputs so positions align with the
    original (uninjected) sequence.

    The enhanced sequence was built as ``orig[:insert_pos] + injected + orig[insert_pos:]``,
    so the logprobs of the original tokens live at ``[0:insert_pos]`` and
    ``[insert_pos + inject_len:]``. The response tokens (after ``insert_pos``) are the
    ones that matter downstream and their logprobs are conditioned on prompt + injected
    turn, which is exactly the privileged conditioning we want. If the teacher output
    is unexpectedly short (e.g. server-side truncation), fall back to truncating to
    ``target_len``, then pad/trim defensively so the caller's shape contract holds.
    """
    if teacher_logprobs_full.shape[0] >= target_len + inject_len:
        teacher_logprobs = torch.cat(
            [teacher_logprobs_full[:insert_pos], teacher_logprobs_full[insert_pos + inject_len:]]
        )
        teacher_ids = torch.cat(
            [teacher_ids_full[:insert_pos], teacher_ids_full[insert_pos + inject_len:]]
        )
    else:
        teacher_logprobs = teacher_logprobs_full[:target_len]
        teacher_ids = teacher_ids_full[:target_len]

    if teacher_ids.shape[0] != target_len:
        diff = target_len - teacher_ids.shape[0]
        if diff > 0:
            teacher_ids = torch.cat([teacher_ids, torch.zeros(diff, dtype=torch.int32)])
            teacher_logprobs = torch.cat([teacher_logprobs, torch.zeros(diff)])
        else:
            teacher_ids = teacher_ids[:target_len]
            teacher_logprobs = teacher_logprobs[:target_len]

    return teacher_ids, teacher_logprobs


# ── Privileged teacher conditioning: one decision point for both teacher paths ──
#
# Two mutually exclusive privileged signals can be injected into the teacher
# context before it scores the student's trajectory:
#
#   OPSD_GOLD_CONDITIONING=1  the sample's ground-truth answer (outcome-independent)
#   HINT_OPD_ENABLED=1        an outcome-conditioned hint that states whether THIS
#                             trajectory was correct (TT-OPD's mechanism)
#
# They are resolved here, in one function, called by both the agent-loop streaming
# path and the batch/colocate path, so the two paths cannot drift apart again.
# Precedence is explicit: gold wins, and enabling both is announced loudly rather
# than silently picking one.
INJECT_GOLD = "gold"
INJECT_HINT_CORRECT = "hint:correct"
INJECT_HINT_INCORRECT = "hint:incorrect"
SKIP_DISABLED = "skip:disabled"
SKIP_MULTIMODAL = "skip:multimodal"
SKIP_NO_TOKENIZER = "skip:no-tokenizer"
SKIP_GOLD_MISSING = "skip:gold-missing"
SKIP_SCORE_MISSING = "skip:score-missing"
SKIP_RENDER_FAILED = "skip:render-failed"

_PRECEDENCE_ANNOUNCED = False


def format_injection_reasons(reasons: dict[str, int]) -> str:
    """Render an injection-reason histogram as one deterministic log line.

    Injected counts are listed first so ``grep 'privileged injection'`` on a training
    log answers "did hints actually fire, and on how many samples" at a glance.
    """
    injected = {k: v for k, v in reasons.items() if not k.startswith("skip:")}
    skipped = {k: v for k, v in reasons.items() if k.startswith("skip:")}
    parts = [f"injected={sum(injected.values())}"]
    parts += [f"{k}={v}" for k, v in sorted(injected.items())]
    parts.append(f"skipped={sum(skipped.values())}")
    parts += [f"{k}={v}" for k, v in sorted(skipped.items())]
    return " ".join(parts)


def hint_correctness(reward_score) -> Optional[bool]:
    """Correctness of one trajectory from its scalar reward, or None if unknowable.

    Returns None — never a guess — when the score is missing, non-numeric or NaN.
    Callers must suppress hint injection in that case: telling the teacher that a
    wrong trajectory was correct is a worse signal than no privileged signal.
    """
    if reward_score is None:
        return None
    try:
        score = float(reward_score)
    except (TypeError, ValueError):
        return None
    if math.isnan(score):
        return None
    return score > _HINT_OPD_CORRECT_THRESHOLD


def resolve_privileged_injection(
    manager: "AsyncTeacherLLMServerManager",
    tokenizer,
    *,
    has_multimodal: bool,
    gold_text: Optional[str] = None,
    reward_score: Optional[float] = None,
) -> tuple[Optional[list[int]], str]:
    """Decide what privileged turn (if any) to splice into the teacher context.

    Returns ``(token_ids_or_None, reason)``. ``reason`` is one of the INJECT_*/SKIP_*
    constants above; it is stable and both the unit tests and the rebuttal checker
    assert on it.

    Gold conditioning takes precedence over outcome-conditioned hints. When gold is
    enabled there is deliberately NO fallback to hints for samples whose ground truth
    is missing: mixing two different privileged signals across the samples of one run
    would make the arm uninterpretable. Those samples are scored without conditioning.
    """
    global _PRECEDENCE_ANNOUNCED
    if _OPSD_GOLD_CONDITIONING and _HINT_OPD_ENABLED and not _PRECEDENCE_ANNOUNCED:
        _PRECEDENCE_ANNOUNCED = True
        print(
            "[Teacher] OPSD_GOLD_CONDITIONING=1 and HINT_OPD_ENABLED=1 are both set. "
            "These are mutually exclusive privileged signals; gold conditioning takes "
            "precedence and outcome-conditioned hints will NOT fire in this run."
        )

    if not (_OPSD_GOLD_CONDITIONING or _HINT_OPD_ENABLED):
        return None, SKIP_DISABLED
    if has_multimodal:
        # Inserting tokens shifts image placeholder positions and breaks VLM processing.
        return None, SKIP_MULTIMODAL
    if tokenizer is None:
        # __init__ raises when conditioning is enabled, so reaching here means a caller
        # passed None explicitly; report it rather than silently scoring unconditioned.
        return None, SKIP_NO_TOKENIZER

    if _OPSD_GOLD_CONDITIONING:
        if not gold_text:
            return None, SKIP_GOLD_MISSING
        gold_ids = manager._build_gold_tokens(gold_text, tokenizer)
        return (gold_ids, INJECT_GOLD) if gold_ids else (None, SKIP_RENDER_FAILED)

    is_correct = hint_correctness(reward_score)
    if is_correct is None:
        return None, SKIP_SCORE_MISSING
    hint_ids = manager._build_hint_tokens(is_correct, tokenizer)
    if not hint_ids:
        return None, SKIP_RENDER_FAILED
    return hint_ids, (INJECT_HINT_CORRECT if is_correct else INJECT_HINT_INCORRECT)


def _resolve_special_token_ids(tokenizer) -> tuple[int, int]:
    """Resolve <|im_start|> and <|im_end|> token IDs from tokenizer."""
    global _IM_START_TOKEN_ID, _IM_END_TOKEN_ID
    if _IM_START_TOKEN_ID is not None and _IM_END_TOKEN_ID is not None:
        return _IM_START_TOKEN_ID, _IM_END_TOKEN_ID

    if tokenizer is not None:
        text_tokenizer = getattr(tokenizer, 'tokenizer', tokenizer)
        try:
            _IM_START_TOKEN_ID = text_tokenizer.convert_tokens_to_ids("<|im_start|>")
            _IM_END_TOKEN_ID = text_tokenizer.convert_tokens_to_ids("<|im_end|>")
            print(f"[Teacher] Resolved im_start={_IM_START_TOKEN_ID}, im_end={_IM_END_TOKEN_ID}")
            return _IM_START_TOKEN_ID, _IM_END_TOKEN_ID
        except Exception:
            pass

    # Fallback: Qwen3.5 defaults
    _IM_START_TOKEN_ID = 248045
    _IM_END_TOKEN_ID = 248046
    return _IM_START_TOKEN_ID, _IM_END_TOKEN_ID

# VLM image special tokens (kept for reference)
# _VISION_TOKEN_IDS = {248053, 248054, 248055, 248056}  # vision_start, vision_end, vision_pad, image_pad


def _get_teacher_sampling_params(
    distillation_config: DistillationConfig,
    distillation_loss_config: DistillationLossConfig,
) -> dict[str, Any]:
    """Get sampling parameters for teacher model when computing log probabilities for distillation."""
    num_logprobs = distillation_loss_config.topk if distillation_loss_config.loss_settings.use_topk else 0
    return {
        "max_tokens": 1,
        "temperature": 0,  # Greedy: uses argmax instead of multinomial, avoids NaN crash
        "prompt_logprobs": num_logprobs,
    }


def _to_ragged_last(
    teacher_ids: torch.Tensor,
    teacher_logprobs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Drop the candidate axis when there is exactly one candidate per position.

    The teacher server returns (S, C): one row per sequence position, C candidates
    wide. C is `prompt_logprobs` from _get_teacher_sampling_params, which is 0 --
    meaning "the actual token only", i.e. C == 1 -- unless top-k distillation is
    enabled.

    That layout cannot survive the trip to the loss. verl transports per-sample
    tensors as nested tensors and requires the RAGGED dim to be LAST
    (verl.utils.tensordict_utils.as_nested_tensor_ragged_last). Here the ragged
    dim is the sequence, and it sits FIRST, so the moment two samples in a batch
    have different lengths -- i.e. essentially always -- chunking the batch dies
    with

        AssertionError: only the last (ragged) dim may vary across samples.
        Got torch.Size([12771, 1]) vs torch.Size([12662, 1])

    raised from _compute_old_log_prob, which names neither this tensor nor this
    module. That is why no distillation arm has ever reached its first optimizer
    step on any backbone.

    With C == 1 the axis carries nothing, so dropping it here makes the tensor
    (S,), which pads to (1, S), batches to (bsz, S) and nests to (bsz, ragged) --
    ragged last, as required. compute_forward_kl_topk puts the axis back before
    gathering; see the matching note there.

    With C > 1 the two requirements genuinely conflict: chunking wants the
    sequence last, and the loss gathers over the candidate axis with dim=-1 so it
    wants candidates last. No single 3-D layout satisfies both, and resolving it
    means changing the transport on both sides. Refuse here, where the shape is
    still explicable, rather than 11 minutes into a cluster job.
    """
    if teacher_logprobs.dim() <= 1:
        return teacher_ids, teacher_logprobs
    n_candidates = teacher_logprobs.shape[-1]
    if n_candidates == 1:
        return teacher_ids.squeeze(-1), teacher_logprobs.squeeze(-1)
    raise NotImplementedError(
        f"teacher logprobs came back {n_candidates} candidates wide, but the nested-tensor "
        "transport requires the ragged (sequence) dim to be last, and the top-k loss gathers "
        "over the candidate dim with dim=-1. Both cannot be last. Top-k distillation needs the "
        "transport layout changed on both sides before it can run; single-candidate "
        "distillation (prompt_logprobs=0) works today."
    )


def _pad_teacher_outputs(
    teacher_ids: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    prompt_width: int,
    response_width: int,
    prompt_length: int,
    response_length: int,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # TODO(wuxibin): remove padding and use tensordict.
    left_pad_size = prompt_width - prompt_length
    right_pad_size = response_width - response_length
    if teacher_ids.ndim == 1:
        # 1D tensors: use 2-element padding (left, right)
        padding = (left_pad_size, right_pad_size)
    else:
        # 2D tensors (topk): pad sequence dim, keep topk dim
        padding = (0, 0, left_pad_size, right_pad_size)
    return (
        F.pad(teacher_ids, padding, value=pad_token_id).unsqueeze(0),
        F.pad(teacher_logprobs, padding, value=0.0).unsqueeze(0),
    )


def _unpad_teacher_inputs(data: DataProto) -> tuple[list[int], int, int]:
    """Unpad valid sequence ids and prompt/response lengths from a single sample.
    The sample is a left-padded prompt concatenated with a right-padded response.
    TODO(wuxibin): remove padding and use tensordict.
    """
    assert len(data) == 1, "Teacher logprob computation expects a single sample"

    input_ids = data.batch["input_ids"][0]
    attention_mask = data.batch["attention_mask"][0]
    prompt_width = data.batch["prompts"][0].shape[0]
    response_width = data.batch["responses"][0].shape[0]
    assert attention_mask.shape[0] == prompt_width + response_width, (
        "attention_mask sequence length must match prompt and response widths"
    )
    valid_prompt_length = int(attention_mask[:prompt_width].sum().item())
    valid_response_length = int(attention_mask[-response_width:].sum().item())
    prompt_num_padding = prompt_width - valid_prompt_length
    sequence_ids = input_ids[prompt_num_padding : prompt_width + valid_response_length]
    sequence_ids = normalize_token_ids(sequence_ids)
    return sequence_ids, valid_prompt_length, valid_response_length


class AsyncTeacherLLMServerManager(AsyncLLMServerManager):
    """Teacher-specific async client used for distillation logprob computation."""

    def __init__(
        self,
        config: DictConfig,
        servers: list[tuple[str, ray.actor.ActorHandle]],
        load_balancer_handle: ray.actor.ActorHandle,
        distillation_config: DictConfig | DistillationConfig,
        pad_token_id: int,
        tokenizer,
    ):
        """``tokenizer`` is REQUIRED — deliberately no default.

        Privileged teacher conditioning renders the injected turn with the model's own
        chat template, so it cannot run without a tokenizer/processor. This parameter
        used to default to None while neither construction site passed one, which made
        every hint silently no-op for the entire v15-v25 run series. A required
        parameter cannot be forgotten, and passing None explicitly while conditioning
        is enabled raises instead of degrading quietly.
        """
        super().__init__(config=config, servers=servers, load_balancer_handle=load_balancer_handle)
        if isinstance(distillation_config, DistillationConfig):
            self.distillation_config = distillation_config
        else:
            self.distillation_config: DistillationConfig = omega_conf_to_dataclass(distillation_config)
        self.distillation_loss_config: DistillationLossConfig = self.distillation_config.distillation_loss
        self.pad_token_id = pad_token_id
        self._tokenizer = tokenizer
        if tokenizer is None:
            message = (
                "AsyncTeacherLLMServerManager was constructed with tokenizer=None. The "
                "injected privileged turn (HINT_OPD_ENABLED / OPSD_GOLD_CONDITIONING) is "
                "rendered with the model's chat template and cannot be built without it."
            )
            if _HINT_OPD_ENABLED or _OPSD_GOLD_CONDITIONING:
                raise ValueError(
                    message + " Privileged conditioning is ENABLED for this run, so refusing "
                    "to start rather than training a mechanism that never fires. Pass the "
                    "tokenizer (or processor) at the construction site."
                )
            print("[Teacher] " + message + " No conditioning is enabled, so this is only a warning.")

    async def compute_teacher_logprobs_single(
        self,
        sequence_ids: list[int],
        multi_modal_data: Optional[dict[str, Any]] = None,
        hint_token_ids: Optional[list[int]] = None,
        original_length: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute teacher log probabilities for a single unpadded sequence.

        If hint_token_ids is provided, injects them before the first assistant turn
        in the sequence, computes teacher logprobs on the enhanced sequence, then
        extracts only the logprobs for the original response tokens. The caller is
        responsible for gating (HINT_OPD_ENABLED or OPSD_GOLD_CONDITIONING): the
        parameter is only non-None when injection is wanted.
        """
        multi_modal_data = multi_modal_data or {}

        if hint_token_ids:
            # Find insertion point: right before the first <|im_start|> in response
            # The sequence is: [prompt tokens] [response tokens]
            # We insert hint as a user message before the response
            enhanced_ids = list(sequence_ids)  # copy
            # Find where response starts (first assistant turn in sequence)
            # For Qwen3.5, response starts at the boundary between prompt and response
            # We insert hint tokens at original_length position (end of prompt)
            if original_length is not None and original_length < len(enhanced_ids):
                enhanced_ids = enhanced_ids[:original_length] + hint_token_ids + enhanced_ids[original_length:]
            else:
                # Fallback: prepend hint before last assistant turn
                enhanced_ids = hint_token_ids + enhanced_ids

            teacher_output = await self.generate(
                request_id=uuid4().hex,
                prompt_ids=enhanced_ids,
                sampling_params=_get_teacher_sampling_params(self.distillation_config, self.distillation_loss_config),
                image_data=multi_modal_data.get("images"),
                video_data=multi_modal_data.get("videos"),
            )
            teacher_ids_full = torch.tensor(teacher_output.extra_fields["prompt_ids"], dtype=torch.int32)
            teacher_logprobs_full = torch.tensor(teacher_output.extra_fields["prompt_logprobs"])

            # Extract logprobs for original sequence positions only.
            # The injected turn sits at original_length, so:
            # enhanced = [0:orig_len] + [injected] + [orig_len:]
            if original_length is not None:
                teacher_ids, teacher_logprobs = _stitch_out_injected_span(
                    teacher_ids_full,
                    teacher_logprobs_full,
                    insert_pos=original_length,
                    inject_len=len(hint_token_ids),
                    target_len=len(sequence_ids),
                )
            else:
                teacher_ids, teacher_logprobs = _stitch_out_injected_span(
                    teacher_ids_full,
                    teacher_logprobs_full,
                    insert_pos=0,
                    inject_len=len(hint_token_ids),
                    target_len=len(sequence_ids),
                )

            return _to_ragged_last(teacher_ids, teacher_logprobs)

        # Standard path (no hints)
        teacher_output = await self.generate(
            request_id=uuid4().hex,
            prompt_ids=sequence_ids,
            sampling_params=_get_teacher_sampling_params(self.distillation_config, self.distillation_loss_config),
            image_data=multi_modal_data.get("images"),
            video_data=multi_modal_data.get("videos"),
        )
        # Shapes: # S, (1 or K), where S is the response length, K is either 1 or topk depending on
        # the distillation loss settings.
        teacher_ids = torch.tensor(teacher_output.extra_fields["prompt_ids"], dtype=torch.int32)
        teacher_logprobs = torch.tensor(teacher_output.extra_fields["prompt_logprobs"])
        assert teacher_ids.shape[0] == teacher_logprobs.shape[0] == len(sequence_ids)
        return _to_ragged_last(teacher_ids, teacher_logprobs)

    def _build_hint_tokens(self, is_correct: bool, tokenizer=None) -> Optional[list[int]]:
        """Build hint token IDs for injection into teacher sequence.

        Renders the hint as one user turn using the model's own chat template, so
        the same code path works for ChatML (Qwen), Gemma, Llama and others. Falls
        back to literal ChatML only if the template cannot be applied.
        """
        if not _HINT_OPD_ENABLED:
            return None
        if tokenizer is None:
            return None

        cached = _HINT_TOKEN_CACHE.get(is_correct)
        if cached is not None:
            return cached

        hint_text = _HINT_CORRECT if is_correct else _HINT_INCORRECT
        hint_ids = _render_user_turn_tokens(hint_text, tokenizer)
        if hint_ids:
            _HINT_TOKEN_CACHE[is_correct] = hint_ids
        return hint_ids

    def _build_gold_tokens(self, gold_text: str, tokenizer=None) -> Optional[list[int]]:
        """Build gold-answer token IDs for OPSD-style privileged teacher conditioning.

        Unlike the two static hints, the gold answer is per-sample; the rendered turn
        is cached keyed by the (truncated) gold text. Uses the same chat-template
        rendering as the hint path so it stays model-family portable, and is spliced
        into the sequence (and stitched back out of the returned logprobs) by exactly
        the same mechanism.
        """
        if not gold_text or tokenizer is None:
            return None

        gold_text = gold_text[:_OPSD_GOLD_MAX_CHARS]
        cached = _GOLD_TOKEN_CACHE.get(gold_text)
        if cached is not None:
            return cached

        # str.replace, not str.format: gold text may itself contain braces.
        message = _OPSD_GOLD_TEMPLATE.replace("{gold}", gold_text)
        gold_ids = _render_user_turn_tokens(message, tokenizer)
        if gold_ids:
            if len(_GOLD_TOKEN_CACHE) >= _GOLD_TOKEN_CACHE_MAX:
                _GOLD_TOKEN_CACHE.clear()
            _GOLD_TOKEN_CACHE[gold_text] = gold_ids
        return gold_ids

    async def compute_teacher_logprobs_batch(self, data: DataProto) -> DataProto:
        """Compute teacher log probabilities for a batch of prompt-response pairs.

        If OPSD_GOLD_CONDITIONING=1, injects the sample's ground-truth answer into
        the teacher sequence (OPSD-style privileged conditioning). Otherwise, if
        HINT_OPD_ENABLED=1, injects the outcome-conditioned hint for this sample's
        trajectory (TT-OPD). Precedence and every skip reason are decided by the
        shared resolve_privileged_injection, the same function the agent-loop
        streaming path calls, so the two paths cannot diverge.
        """
        multi_modal_data_batch = data.non_tensor_batch.get("teacher_multi_modal_data")
        tasks = []
        lengths = []
        prompt_width = data.batch["prompts"].shape[1]
        response_width = data.batch["responses"].shape[1]

        # Trajectory-level reward, used to pick the hint direction. rm_scores is set by
        # extract_reward() before teacher logprob computation; shape [bsz, seq_len], so
        # sum over the sequence to recover the scalar trajectory reward. Absent here
        # means the score is unknown and hints are suppressed (never guessed).
        reward_scores = data.batch["rm_scores"].sum(dim=-1) if "rm_scores" in data.batch else None

        tokenizer = self._tokenizer
        reasons: dict[str, int] = {}
        # Compute logprobs for each sample in the batch
        for i in range(len(data)):
            item = data[i : i + 1]
            sequence_ids, prompt_length, response_length = _unpad_teacher_inputs(item)
            multi_modal_data = None if multi_modal_data_batch is None else multi_modal_data_batch[i]
            lengths.append((prompt_length, response_length))

            has_multimodal = bool(
                multi_modal_data is not None
                and (multi_modal_data.get("images") or multi_modal_data.get("videos"))
            )
            hint_token_ids, reason = resolve_privileged_injection(
                self,
                tokenizer,
                has_multimodal=has_multimodal,
                gold_text=_extract_gold_text(data, i) if _OPSD_GOLD_CONDITIONING else None,
                reward_score=None if reward_scores is None else reward_scores[i],
            )
            reasons[reason] = reasons.get(reason, 0) + 1

            # Pass image/video data to teacher for VLM-aware logprob computation.
            # return_exceptions=True in asyncio.gather provides fallback if alignment fails.
            tasks.append(
                asyncio.create_task(
                    self.compute_teacher_logprobs_single(
                        sequence_ids=sequence_ids,
                        multi_modal_data=multi_modal_data,
                        hint_token_ids=hint_token_ids,
                        original_length=prompt_length if hint_token_ids else None,
                    )
                )
            )

        if _OPSD_GOLD_CONDITIONING or _HINT_OPD_ENABLED:
            print(f"[Teacher-batch] privileged injection over {len(data)} samples: " + format_injection_reasons(reasons))

        # Use return_exceptions to avoid crashing the entire batch on a single failure
        outputs_raw = await asyncio.gather(*tasks, return_exceptions=True)
        outputs = []
        for idx, result in enumerate(outputs_raw):
            if isinstance(result, Exception):
                # Failed sample — use zero logprobs as fallback
                prompt_length, response_length = lengths[idx]
                seq_len = prompt_length + response_length
                fallback_ids = torch.zeros(seq_len, dtype=torch.int32)
                fallback_logprobs = torch.zeros(seq_len)
                outputs.append((fallback_ids, fallback_logprobs))
                print(f"[Teacher] Sample {idx} failed: {type(result).__name__}: {result}")
            else:
                outputs.append(result)

        # Pad the teacher logprobs and ids
        padded_teacher_ids = []
        padded_teacher_logprobs = []
        for (teacher_ids, teacher_logprobs), (prompt_length, response_length) in zip(outputs, lengths, strict=True):
            padded_ids, padded_logprobs = _pad_teacher_outputs(
                teacher_ids,
                teacher_logprobs,
                prompt_width=prompt_width,
                response_width=response_width,
                prompt_length=prompt_length,
                response_length=response_length,
                pad_token_id=self.pad_token_id,
            )
            padded_teacher_ids.append(padded_ids)
            padded_teacher_logprobs.append(padded_logprobs)

        batch = TensorDict(
            {
                "teacher_ids": torch.cat(padded_teacher_ids),
                "teacher_logprobs": torch.cat(padded_teacher_logprobs),
            },
            batch_size=len(data),
        )
        return DataProto(batch=batch)
