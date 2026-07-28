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

"""CPU unit tests for the OPSD baseline distillation loss (loss_mode="opsd").

Covers: registration, correct masking, NaN/inf sanitization, per-token clipping
(OPSD's token_clip) bounding contributions, zero loss on zero divergence, and
that TT-OPD's turn truncation / reward-sign flip are NOT applied.
"""

import os

import pytest
import torch
from tensordict import TensorDict

from verl.trainer.distillation.losses import (
    compute_opsd_loss,
    get_distillation_loss_fn,
    get_distillation_loss_settings,
)

PROMPT_LEN = 2
RESP_LEN = 4
BSZ = 2


def _flat_from_padded(padded_resp_values: torch.Tensor, prompt_fill: float = -1.0) -> torch.Tensor:
    """Build a no-padding flat tensor whose no_padding_2_padding output equals
    ``padded_resp_values`` ([BSZ, RESP_LEN]) for fully-valid sequences.

    no_padding_2_padding slices values[seq_offset - resp_len - 1 : seq_offset - 1]
    (left-shift by one), so response position j maps to flat index
    (seq_offset - resp_len - 1 + j).
    """
    seq_len = PROMPT_LEN + RESP_LEN
    flat = torch.full((BSZ * seq_len,), prompt_fill, dtype=torch.float32)
    for i in range(BSZ):
        seq_offset = (i + 1) * seq_len
        start = seq_offset - RESP_LEN - 1
        flat[start : start + RESP_LEN] = padded_resp_values[i]
    return flat


def _make_data(
    teacher_padded: torch.Tensor,
    response_mask: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
) -> TensorDict:
    if response_mask is None:
        response_mask = torch.ones(BSZ, RESP_LEN)
    seq_len = PROMPT_LEN + RESP_LEN
    entries = {
        "prompts": torch.ones(BSZ, PROMPT_LEN, dtype=torch.long),
        "responses": torch.ones(BSZ, RESP_LEN, dtype=torch.long),
        "attention_mask": torch.ones(BSZ, seq_len, dtype=torch.long),
        "response_mask": response_mask,
        "teacher_logprobs": _flat_from_padded(teacher_padded),
    }
    if advantages is not None:
        entries["advantages"] = advantages
    return TensorDict(entries, batch_size=[])


def _model_output(student_padded: torch.Tensor) -> dict:
    return {"log_probs": _flat_from_padded(student_padded)}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("OPSD_TOKEN_CLIP", raising=False)
    yield


def test_opsd_is_registered():
    fn = get_distillation_loss_fn("opsd")
    assert fn is compute_opsd_loss
    settings = get_distillation_loss_settings("opsd")
    assert settings.use_estimator and not settings.use_topk


def test_zero_divergence_yields_zero_loss():
    logprobs = torch.full((BSZ, RESP_LEN), -1.5)
    losses, metrics = compute_opsd_loss(None, None, _model_output(logprobs), _make_data(logprobs.clone()))
    assert losses.shape == (BSZ, RESP_LEN)
    assert torch.allclose(losses, torch.zeros_like(losses), atol=1e-7)
    assert metrics["opsd/kl_mean"].aggregate() == pytest.approx(0.0, abs=1e-7)
    assert metrics["opsd/kl_max"].aggregate() == pytest.approx(0.0, abs=1e-7)


def test_k3_value_matches_closed_form():
    student = torch.full((BSZ, RESP_LEN), -1.0)
    teacher = torch.full((BSZ, RESP_LEN), -2.0)
    losses, metrics = compute_opsd_loss(None, None, _model_output(student), _make_data(teacher))
    # k3 = exp(t - s) - (t - s) - 1 with t - s = -1
    expected = torch.exp(torch.tensor(-1.0)) - (-1.0) - 1.0
    assert torch.allclose(losses, torch.full_like(losses, expected.item()), atol=1e-6)
    assert (losses >= 0).all(), "k3 estimator must be non-negative"
    # signed k1 = s - t = +1 (student more confident than teacher)
    assert metrics["opsd/k1_signed_mean"].aggregate() == pytest.approx(1.0, abs=1e-6)


def test_masking_metrics_use_only_valid_tokens():
    student = torch.full((BSZ, RESP_LEN), -1.0)
    teacher = torch.full((BSZ, RESP_LEN), -1.0)
    # Put a huge divergence ONLY at masked-out positions.
    teacher[:, -1] = -15.0
    mask = torch.ones(BSZ, RESP_LEN)
    mask[:, -1] = 0.0
    losses, metrics = compute_opsd_loss(None, None, _model_output(student), _make_data(teacher, response_mask=mask))
    # Metrics must ignore the masked-out positions entirely.
    assert metrics["opsd/kl_mean"].aggregate() == pytest.approx(0.0, abs=1e-6)
    assert metrics["opsd/kl_max"].aggregate() == pytest.approx(0.0, abs=1e-6)
    # The framework aggregates the per-token matrix against response_mask; the
    # masked aggregate must be unaffected by the poisoned positions.
    masked_sum = (losses * mask).sum()
    assert masked_sum == pytest.approx(0.0, abs=1e-6)
    # Unmasked positions genuinely diverge (sanity that the poison was real).
    assert losses[:, -1].min() > 1.0


def test_nan_inf_sanitization():
    student = torch.full((BSZ, RESP_LEN), -1.0)
    teacher = torch.full((BSZ, RESP_LEN), -1.0)
    teacher[:, 0] = float("nan")  # first-token NaN, as produced by the teacher server
    teacher[0, 1] = float("-inf")  # zero-probability token
    student[1, 2] = float("-inf")
    losses, metrics = compute_opsd_loss(None, None, _model_output(student), _make_data(teacher))
    assert torch.isfinite(losses).all(), "loss must be finite after sanitization"
    for key in ("opsd/kl_mean", "opsd/kl_p99", "opsd/kl_max"):
        assert torch.isfinite(torch.tensor(metrics[key].aggregate())), f"{key} must be finite"
    # NaN maps to -20 (not 0): the sanitized first token must show non-zero divergence
    # against the -1.0 student, i.e. the biasing-toward-prob-1.0 bug is absent.
    assert losses[:, 0].min() > 0.1


def test_token_clip_bounds_per_token_contributions(monkeypatch):
    student = torch.full((BSZ, RESP_LEN), -1.0)
    teacher = torch.full((BSZ, RESP_LEN), -1.0)
    teacher[:, 1] = -9.0  # k3(t-s=-8) ≈ 7.0 — a style-token-like KL spike

    # Clipping OFF by default: the spike passes through.
    losses_unclipped, metrics_unclipped = compute_opsd_loss(
        None, None, _model_output(student), _make_data(teacher)
    )
    assert losses_unclipped[:, 1].min() > 5.0
    assert metrics_unclipped["opsd/clip_frac"].aggregate() == pytest.approx(0.0)
    assert metrics_unclipped["opsd/token_clip"] == 0.0

    # Clipping ON: every per-token contribution is bounded by the clip value.
    monkeypatch.setenv("OPSD_TOKEN_CLIP", "0.05")
    losses_clipped, metrics_clipped = compute_opsd_loss(
        None, None, _model_output(student), _make_data(teacher)
    )
    assert losses_clipped.max() <= 0.05 + 1e-8
    assert metrics_clipped["opsd/token_clip"] == pytest.approx(0.05)
    # Exactly one of RESP_LEN positions per sequence exceeded the clip.
    assert metrics_clipped["opsd/clip_frac"].aggregate() == pytest.approx(1.0 / RESP_LEN, abs=1e-6)


def test_token_clip_zeroes_gradient_at_clipped_tokens(monkeypatch):
    monkeypatch.setenv("OPSD_TOKEN_CLIP", "0.05")
    student_padded = torch.full((BSZ, RESP_LEN), -1.0)
    teacher = torch.full((BSZ, RESP_LEN), -1.0)
    teacher[:, 1] = -9.0
    student_flat = _flat_from_padded(student_padded).requires_grad_(True)
    losses, _ = compute_opsd_loss(None, None, {"log_probs": student_flat}, _make_data(teacher))
    losses.sum().backward()
    grad_padded = torch.stack(
        [
            student_flat.grad[(i + 1) * (PROMPT_LEN + RESP_LEN) - RESP_LEN - 1 :][:RESP_LEN]
            for i in range(BSZ)
        ]
    )
    # Clipped (spiked) position contributes zero gradient; unclipped near-zero-KL
    # positions have (tiny but) untouched gradient paths.
    assert torch.allclose(grad_padded[:, 1], torch.zeros(BSZ), atol=1e-9)


def test_no_reward_sign_flip_and_no_truncation():
    """TT-OPD's bidirectional flip and turn truncation must NOT apply in opsd mode."""
    student = torch.full((BSZ, RESP_LEN), -1.0)
    teacher = torch.full((BSZ, RESP_LEN), -2.0)
    negative_adv = torch.full((BSZ, RESP_LEN), -1.0)
    losses_with_adv, _ = compute_opsd_loss(
        None, None, _model_output(student), _make_data(teacher, advantages=negative_adv)
    )
    losses_without_adv, _ = compute_opsd_loss(None, None, _model_output(student), _make_data(teacher))
    # Negative-advantage trajectories are NOT flipped: identical, all non-negative.
    assert torch.equal(losses_with_adv, losses_without_adv)
    assert (losses_with_adv >= 0).all()
    # No turn truncation: every valid position carries loss even with BT_OPD_MAX_TURN set.
    prev = os.environ.get("BT_OPD_MAX_TURN")
    os.environ["BT_OPD_MAX_TURN"] = "1"
    try:
        losses_turn, _ = compute_opsd_loss(None, None, _model_output(student), _make_data(teacher))
    finally:
        if prev is None:
            os.environ.pop("BT_OPD_MAX_TURN", None)
        else:
            os.environ["BT_OPD_MAX_TURN"] = prev
    assert (losses_turn > 0).all()


def test_kl_distribution_metrics_present_and_ordered():
    torch.manual_seed(0)
    student = -torch.rand(BSZ, RESP_LEN) - 0.5
    teacher = -torch.rand(BSZ, RESP_LEN) - 0.5
    _, metrics = compute_opsd_loss(None, None, _model_output(student), _make_data(teacher))
    p50 = metrics["opsd/kl_p50"].aggregate()
    p90 = metrics["opsd/kl_p90"].aggregate()
    p99 = metrics["opsd/kl_p99"].aggregate()
    kl_max = metrics["opsd/kl_max"].aggregate()
    assert p50 <= p90 <= p99 <= kl_max + 1e-8
