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
        tokenizer=None,
    ):
        super().__init__(config=config, servers=servers, load_balancer_handle=load_balancer_handle)
        if isinstance(distillation_config, DistillationConfig):
            self.distillation_config = distillation_config
        else:
            self.distillation_config: DistillationConfig = omega_conf_to_dataclass(distillation_config)
        self.distillation_loss_config: DistillationLossConfig = self.distillation_config.distillation_loss
        self.pad_token_id = pad_token_id
        self._tokenizer = tokenizer

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
        extracts only the logprobs for the original response tokens.
        """
        multi_modal_data = multi_modal_data or {}

        if hint_token_ids and _HINT_OPD_ENABLED:
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

            # Extract logprobs for original sequence positions only
            # The hint was inserted at original_length, so:
            # enhanced = [0:orig_len] + [hint] + [orig_len:]
            # We need logprobs at positions [0:orig_len] and [orig_len+hint_len:]
            hint_len = len(hint_token_ids)
            if original_length is not None and teacher_logprobs_full.shape[0] >= len(sequence_ids) + hint_len:
                # Stitch: prompt logprobs (before hint) + response logprobs (after hint)
                pre_hint = teacher_logprobs_full[:original_length]
                post_hint = teacher_logprobs_full[original_length + hint_len:]
                teacher_logprobs = torch.cat([pre_hint, post_hint])
                pre_ids = teacher_ids_full[:original_length]
                post_ids = teacher_ids_full[original_length + hint_len:]
                teacher_ids = torch.cat([pre_ids, post_ids])
            else:
                # Fallback: truncate to original length
                teacher_logprobs = teacher_logprobs_full[:len(sequence_ids)]
                teacher_ids = teacher_ids_full[:len(sequence_ids)]

            # Ensure length matches
            if teacher_ids.shape[0] != len(sequence_ids):
                # Pad or truncate
                diff = len(sequence_ids) - teacher_ids.shape[0]
                if diff > 0:
                    teacher_ids = torch.cat([teacher_ids, torch.zeros(diff, dtype=torch.int32)])
                    teacher_logprobs = torch.cat([teacher_logprobs, torch.zeros(diff)])
                else:
                    teacher_ids = teacher_ids[:len(sequence_ids)]
                    teacher_logprobs = teacher_logprobs[:len(sequence_ids)]

            return teacher_ids, teacher_logprobs

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
        return teacher_ids, teacher_logprobs

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
        # Handle VLM processors (e.g., Qwen3VLProcessor) which wrap a text tokenizer
        text_tokenizer = getattr(tokenizer, 'tokenizer', tokenizer)

        hint_ids: Optional[list[int]] = None
        try:
            rendered = text_tokenizer.apply_chat_template(
                [{"role": "user", "content": hint_text}],
                tokenize=False,
                add_generation_prompt=False,
            )
            # The hint is spliced into the middle of an existing sequence, so any
            # sequence-start marker the template prepends must be dropped.
            bos = getattr(text_tokenizer, "bos_token", None)
            if bos and rendered.startswith(bos):
                rendered = rendered[len(bos):]
            hint_ids = text_tokenizer.encode(rendered, add_special_tokens=False)
        except Exception as exc:
            print(f"[Teacher] chat template unavailable for hint ({exc}); using ChatML fallback")

        if not hint_ids:
            content_ids = text_tokenizer.encode(f"user\n{hint_text}", add_special_tokens=False)
            im_start, im_end = _resolve_special_token_ids(text_tokenizer)
            newline_ids = text_tokenizer.encode("\n", add_special_tokens=False)
            hint_ids = [im_start] + content_ids + [im_end] + newline_ids

        _HINT_TOKEN_CACHE[is_correct] = hint_ids
        return hint_ids

    async def compute_teacher_logprobs_batch(self, data: DataProto) -> DataProto:
        """Compute teacher log probabilities for a batch of prompt-response pairs.

        If HINT_OPD_ENABLED=1, injects reward-based hints into teacher sequences
        before computing logprobs (Bidirectional Hint-OPD).
        """
        multi_modal_data_batch = data.non_tensor_batch.get("teacher_multi_modal_data")
        tasks = []
        lengths = []
        prompt_width = data.batch["prompts"].shape[1]
        response_width = data.batch["responses"].shape[1]

        # Check for reward data to determine hint direction
        # rm_scores is set by extract_reward() before teacher logprob computation
        has_rewards = "rm_scores" in data.batch
        reward_scores = None
        if has_rewards and _HINT_OPD_ENABLED:
            # rm_scores shape: [batch_size, seq_len] — sum to get trajectory reward
            reward_scores = data.batch["rm_scores"].sum(dim=-1)

        # Get tokenizer for hint encoding
        tokenizer = getattr(self, '_tokenizer', None)
        if tokenizer is None and hasattr(self, 'config') and hasattr(self.config, 'tokenizer'):
            tokenizer = self.config.tokenizer

        num_with_hints = 0
        # Compute logprobs for each sample in the batch
        for i in range(len(data)):
            item = data[i : i + 1]
            sequence_ids, prompt_length, response_length = _unpad_teacher_inputs(item)
            multi_modal_data = None if multi_modal_data_batch is None else multi_modal_data_batch[i]
            lengths.append((prompt_length, response_length))

            # Build hint tokens if reward data available
            # Skip hint injection for samples with images/videos — inserting tokens
            # shifts image placeholder positions and breaks VLM processing
            hint_token_ids = None
            has_multimodal = multi_modal_data is not None and (
                multi_modal_data.get("images") or multi_modal_data.get("videos")
            )
            if reward_scores is not None and _HINT_OPD_ENABLED and tokenizer is not None and not has_multimodal:
                is_correct = float(reward_scores[i]) > 0
                hint_token_ids = self._build_hint_tokens(is_correct, tokenizer)
                if hint_token_ids:
                    num_with_hints += 1

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

        if _HINT_OPD_ENABLED:
            num_skipped_mm = len(data) - num_with_hints
            print(f"[Hint-OPD] Injected hints into {num_with_hints}/{len(data)} samples (skipped {num_skipped_mm} multimodal)")

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
