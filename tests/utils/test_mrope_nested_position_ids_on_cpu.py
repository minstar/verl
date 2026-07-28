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

"""CPU regression tests for the ragged dimension of 3D (mRoPE) ``position_ids``.

A VLM such as Qwen3.5 carries ``position_ids`` of shape ``(bsz, 4, seq_len)``. verl moves
that into a jagged NestedTensor whose ragged dim must be the *sequence* dim. It used to be
built with ``torch.nested.as_nested_tensor()``, which does not take the ragged dim as an
argument -- it infers it by looking for the dim whose size varies across the batch, and
falls back to dim 1 when every sample has an identical shape. So a group whose sequence
lengths happened to collide (two rollouts of the same prompt that both hit
``max_response_length``, or a group of size one) produced a tensor ragged over the
*mRoPE-section* dim, with ``offsets()`` counting 4 sections instead of tokens.

Downstream that turned into ``position_ids`` of shape ``(4, 1, 8)`` alongside 4483 packed
tokens, and the model died in ``apply_rotary_pos_emb``::

    RuntimeError: The size of tensor a (4483) must match the size of tensor b (8)
                  at non-singleton dimension 2

The failure is data dependent -- training survived the steps where the seqlen-balanced
partition happened to give every rank two differently-sized sequences.
"""

import pickle

import pytest
import torch
from tensordict import TensorDict

from verl.utils.tensordict_utils import (
    as_nested_tensor_ragged_last,
    chunk_tensordict,
    contiguous,
    maybe_fix_3d_position_ids,
)
from verl.workers.utils.padding import left_right_2_no_padding

N_MROPE_SECTIONS = 4
MAX_PROMPT_LEN = 64
MAX_RESPONSE_LEN = 128


def _make_left_right_padded_batch(lengths: list[tuple[int, int]]) -> TensorDict:
    """Build the left-right padded TensorDict the agent loop hands to the trainer.

    Args:
        lengths: one ``(prompt_len, response_len)`` per sample.
    """
    bsz = len(lengths)
    seq_len = MAX_PROMPT_LEN + MAX_RESPONSE_LEN
    input_ids = torch.zeros(bsz, seq_len, dtype=torch.long)
    attention_mask = torch.zeros(bsz, seq_len, dtype=torch.long)
    response_mask = torch.zeros(bsz, MAX_RESPONSE_LEN, dtype=torch.long)
    position_ids = torch.zeros(bsz, N_MROPE_SECTIONS, seq_len, dtype=torch.long)

    for i, (prompt_len, response_len) in enumerate(lengths):
        start, end = MAX_PROMPT_LEN - prompt_len, MAX_PROMPT_LEN + response_len
        attention_mask[i, start:end] = 1
        input_ids[i, start:end] = torch.randint(1, 1000, (prompt_len + response_len,))
        response_mask[i, :response_len] = 1
        valid = attention_mask[i].bool()
        for section in range(N_MROPE_SECTIONS):
            # a distinct ramp per section so a section/sequence mix-up is visible in the values
            position_ids[i, section, valid] = torch.arange(prompt_len + response_len) + section

    return TensorDict(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "response_mask": response_mask,
            "position_ids": position_ids,
            "prompts": input_ids[:, :MAX_PROMPT_LEN],
            "responses": input_ids[:, MAX_PROMPT_LEN:],
        },
        batch_size=bsz,
    )


def _expected_valid_position_ids(td: TensorDict, index: int) -> torch.Tensor:
    return td["position_ids"][index][:, td["attention_mask"][index].bool()]


def _model_inputs(micro_batch: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
    """The shapes ``FSDPEngineWithLMHead.prepare_model_inputs`` derives for the packed path."""
    input_ids, position_ids = micro_batch["input_ids"], micro_batch["position_ids"]
    input_ids_rmpad = input_ids.values().unsqueeze(0)  # (1, total_nnz)
    if position_ids.dim() == 3:
        position_ids_rmpad = position_ids.values().unsqueeze(1)  # (4, 1, total_nnz)
    else:
        position_ids_rmpad = position_ids.values().unsqueeze(0)  # (1, total_nnz)
    return input_ids_rmpad, position_ids_rmpad


# --------------------------------------------------------------------------------------
# the helper itself
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "seq_lens",
    [
        pytest.param([7, 5, 3], id="distinct_lengths"),
        pytest.param([5, 5, 5], id="colliding_lengths"),
        pytest.param([5], id="single_sample"),
        pytest.param([5, 5, 3, 3], id="pairwise_colliding"),
    ],
)
def test_as_nested_tensor_ragged_last_layout_is_independent_of_lengths(seq_lens):
    components = [torch.arange(N_MROPE_SECTIONS * n).view(N_MROPE_SECTIONS, n) for n in seq_lens]
    nested = as_nested_tensor_ragged_last(components)

    assert nested._ragged_idx == 2, "ragged dim must be the sequence dim regardless of the lengths"
    assert nested.values().shape == (N_MROPE_SECTIONS, sum(seq_lens))
    assert nested.offsets().tolist() == [0, *torch.tensor(seq_lens).cumsum(0).tolist()]
    assert nested.is_contiguous()
    for got, want in zip(nested.unbind(), components, strict=True):
        assert torch.equal(got, want)

    # torch's inference is what this helper exists to avoid: it silently falls back to the
    # section dim as soon as every sample in the group has the same shape
    inferred = torch.nested.as_nested_tensor(components, layout=torch.jagged)
    if len(set(seq_lens)) == 1:
        assert inferred._ragged_idx == 1, "guard: torch still mis-infers, so the helper is still needed"
    else:
        assert inferred._ragged_idx == 2


# --------------------------------------------------------------------------------------
# construction inside the trainer
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "lengths",
    [
        pytest.param([(20, 100), (30, 80)], id="distinct_lengths"),  # totals 120 / 110
        # two rollouts of the same prompt, both clipped at max_response_length -> identical totals
        pytest.param([(20, 128), (20, 128)], id="colliding_lengths"),
        pytest.param([(20, 128)], id="single_sample"),
    ],
)
def test_left_right_2_no_padding_pins_ragged_dim_to_sequence(lengths):
    source = _make_left_right_padded_batch(lengths)
    td = left_right_2_no_padding(source.clone())

    input_ids, position_ids = td["input_ids"], td["position_ids"]
    total_tokens = sum(p + r for p, r in lengths)

    assert position_ids.dim() == 3
    assert position_ids._ragged_idx == 2
    assert position_ids.values().shape == (N_MROPE_SECTIONS, total_tokens)
    # position_ids must be cut exactly like input_ids, which is what prepare_model_inputs relies on
    assert torch.equal(position_ids.offsets(), input_ids.offsets().to(position_ids.offsets().dtype))
    assert input_ids.values().shape[0] == total_tokens

    for i in range(len(lengths)):
        assert torch.equal(position_ids.unbind()[i], _expected_valid_position_ids(source, i))


# --------------------------------------------------------------------------------------
# the full trainer -> worker -> micro-batch path that actually crashed
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "lengths",
    [
        pytest.param([(20, 100), (30, 80), (25, 70), (40, 60)], id="distinct_lengths"),  # 120/110/95/100
        # every DP rank below receives two identically sized sequences: the crashing case
        pytest.param([(20, 128), (20, 128), (30, 128), (30, 128)], id="colliding_lengths"),
    ],
)
def test_dp_dispatch_and_micro_batching_keep_positions_aligned(lengths):
    """trainer batch -> DP chunk -> ray round trip -> micro-batch -> model inputs."""
    source = _make_left_right_padded_batch(lengths)
    td = left_right_2_no_padding(source.clone())

    dp_size = len(lengths) // 2  # two sequences per rank, as in the failing run
    sample_idx = 0
    for rank_td in chunk_tensordict(td, dp_size):
        # what the ray dispatcher does before shipping the TensorDict to the worker
        rank_td = pickle.loads(pickle.dumps(contiguous(rank_td).consolidate()))
        # first thing the engine does on the worker side
        maybe_fix_3d_position_ids(rank_td)
        assert rank_td["position_ids"]._ragged_idx == 2

        # prepare_micro_batches with use_dynamic_bsz=False and micro_batch_size_per_gpu=1
        for micro_batch in chunk_tensordict(rank_td, 2):
            input_ids_rmpad, position_ids_rmpad = _model_inputs(micro_batch)

            assert position_ids_rmpad.shape[0] == N_MROPE_SECTIONS
            assert position_ids_rmpad.shape[1] == 1
            assert position_ids_rmpad.shape[-1] == input_ids_rmpad.shape[-1], (
                "packed position_ids and packed input_ids must cover the same number of tokens"
            )
            expected = _expected_valid_position_ids(source, sample_idx)
            assert torch.equal(position_ids_rmpad[:, 0, :], expected), "position id values were reordered"
            sample_idx += 1

    assert sample_idx == len(lengths)


# --------------------------------------------------------------------------------------
# the padded (use_remove_padding=False) branch of prepare_model_inputs
# --------------------------------------------------------------------------------------


def test_attention_mask_covers_the_whole_sequence_not_just_the_response():
    """``use_remove_padding=False`` must not mask out the tail of a sequence.

    ``prepare_model_inputs`` used to build the padded branch's attention mask from
    ``loss_mask``, which ``left_right_2_no_padding`` aliases to the *response* mask
    (length ``max_response_length``). Padding that up to the full sequence length marked
    the last ``seq_len - max_response_length`` tokens -- the tail of the response -- as
    padding on every sample.
    """
    from verl.workers.engine.utils import attention_mask_from_seq_lens

    lengths = [(20, 128), (30, 100)]
    td = left_right_2_no_padding(_make_left_right_padded_batch(lengths))

    input_ids = td["input_ids"]
    seq_len_effective = input_ids.offsets().diff()
    max_seq_len = max(seq_len_effective)
    assert seq_len_effective.tolist() == [p + r for p, r in lengths]

    attention_mask = attention_mask_from_seq_lens(seq_len_effective, max_seq_len)
    assert attention_mask.shape == (len(lengths), int(max_seq_len))
    assert attention_mask.sum(dim=1).tolist() == seq_len_effective.tolist()
    for i, total in enumerate(seq_len_effective.tolist()):
        assert attention_mask[i, :total].all(), "real tokens must be visible"
        assert not attention_mask[i, total:].any(), "padding must be masked"

    # the old derivation, kept here so the regression is legible
    stale = torch.nested.to_padded_tensor(
        torch.nested.as_nested_tensor(
            [torch.ones_like(t, dtype=torch.int32) for t in td["loss_mask"]], layout=torch.jagged
        ),
        padding=0,
        output_size=(len(lengths), max_seq_len),
    )
    assert stale.sum(dim=1).tolist() == [MAX_RESPONSE_LEN] * len(lengths)
    assert stale.sum(dim=1).tolist() != seq_len_effective.tolist()


# --------------------------------------------------------------------------------------
# drive the model code that raised the original error
# --------------------------------------------------------------------------------------


def _tiny_qwen3_5_text_config():
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    return Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        layer_types=["full_attention", "full_attention"],
        max_position_embeddings=512,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "mrope_section": [2, 2, 2],
            "mrope_interleaved": True,
            "partial_rotary_factor": 0.75,
        },
        attn_implementation="sdpa",
    )


def test_packed_mrope_positions_are_accepted_by_qwen3_5_forward():
    """The packed inputs the fixed pipeline produces must run through the real model code."""
    pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

    lengths = [(20, 128), (20, 128)]  # colliding lengths: the case that used to crash
    td = left_right_2_no_padding(_make_left_right_padded_batch(lengths))
    micro_batch = chunk_tensordict(td, 2)[0]
    input_ids_rmpad, position_ids_rmpad = _model_inputs(micro_batch)
    input_ids_rmpad = input_ids_rmpad % 64  # fit the tiny vocab

    model = Qwen3_5TextModel(_tiny_qwen3_5_text_config()).eval()
    with torch.no_grad():
        out = model(
            input_ids=input_ids_rmpad,
            position_ids=position_ids_rmpad,
            attention_mask=None,
            use_cache=False,
        )
    assert out.last_hidden_state.shape[:2] == input_ids_rmpad.shape


def test_section_ragged_position_ids_reproduce_the_reported_rotary_error():
    """Negative control: the pre-fix layout still fails, so this test can detect a regression."""
    pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5TextRotaryEmbedding,
        apply_rotary_pos_emb,
    )

    config = _tiny_qwen3_5_text_config()
    rotary = Qwen3_5TextRotaryEmbedding(config)
    total_tokens, batch_size = 4483, 2
    rotary_dim = int(config.head_dim * config.rope_parameters["partial_rotary_factor"])

    def cos_sin(position_ids):
        # Qwen3_5TextModel.forward strips the text section before calling the rotary embedding
        if position_ids.ndim == 3 and position_ids.shape[0] == N_MROPE_SECTIONS:
            position_ids = position_ids[1:]
        return rotary(torch.zeros(1, total_tokens, config.hidden_size), position_ids)

    query = torch.zeros(1, config.num_attention_heads, total_tokens, config.head_dim)
    key = torch.zeros(1, config.num_key_value_heads, total_tokens, config.head_dim)

    good_cos, good_sin = cos_sin(torch.zeros(N_MROPE_SECTIONS, 1, total_tokens, dtype=torch.long))
    assert good_cos.shape == (1, total_tokens, rotary_dim)
    apply_rotary_pos_emb(query, key, good_cos, good_sin)  # must not raise

    # what a section-ragged nested tensor degrades into: offsets count sections, not tokens
    bogus_len = N_MROPE_SECTIONS * batch_size
    bad_cos, bad_sin = cos_sin(torch.zeros(N_MROPE_SECTIONS, 1, bogus_len, dtype=torch.long))
    with pytest.raises(RuntimeError, match=rf"tensor a \({total_tokens}\).*tensor b \({bogus_len}\)"):
        apply_rotary_pos_emb(query, key, bad_cos, bad_sin)
