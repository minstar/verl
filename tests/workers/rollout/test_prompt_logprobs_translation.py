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
"""sglang's input logprobs, in vLLM's `prompt_logprobs` layout.

`prompt_logprobs` is a vLLM SamplingParams field. sglang rejects it outright
(`TypeError: SamplingParams.__init__() got an unexpected keyword argument
'prompt_logprobs'`), which is why the distillation teacher could not run against
an sglang server at all. sglang does return the same information under
`meta_info["input_token_logprobs"]` / `["input_top_logprobs"]`, in a different
shape.

The layout is load-bearing: teacher_manager asserts one row per prompt token and
feeds the values straight into the distillation KL. A one-position shift would
not raise anywhere -- it would just make every teacher logprob describe the wrong
token. So alignment is asserted, and a wrong row count must raise rather than be
sliced into shape.
"""

import pytest

from verl.workers.rollout.sglang_rollout.async_sglang_server import prompt_logprobs_from_meta


def entry(logprob, token_id):
    """One sglang logprob entry: (logprob, token_id, text)."""
    return (logprob, token_id, None)


def meta_topk(rows):
    return {"input_top_logprobs": rows}


def meta_flat(entries):
    return {"input_token_logprobs": entries}


# A 4-token prompt [10, 11, 12, 13]. Token 0 has no predecessor, so sglang
# reports logprobs for tokens 1..3 only.
FLAT_3 = [entry(-0.1, 11), entry(-0.2, 12), entry(-0.3, 13)]


def test_flat_case_has_one_row_per_prompt_token():
    lp, ids = prompt_logprobs_from_meta(meta_flat(FLAT_3), 0, prompt_len=4)
    assert len(lp) == 4 and len(ids) == 4


def test_row_i_describes_token_i_plus_one():
    lp, ids = prompt_logprobs_from_meta(meta_flat(FLAT_3), 0, prompt_len=4)
    assert ids[:3] == [[11], [12], [13]]
    assert lp[:3] == [[-0.1], [-0.2], [-0.3]]


def test_last_row_is_a_dummy_because_nothing_follows_it():
    lp, ids = prompt_logprobs_from_meta(meta_flat(FLAT_3), 0, prompt_len=4)
    assert lp[-1] == [0.0] and ids[-1] == [0]


def test_leading_placeholder_for_the_first_token_is_dropped():
    """sglang may emit a None-logprob row for token 0; it must not shift the rest."""
    with_placeholder = [entry(None, 10)] + FLAT_3
    lp, ids = prompt_logprobs_from_meta(meta_flat(with_placeholder), 0, prompt_len=4)
    assert ids[:3] == [[11], [12], [13]]


def test_topk_rows_keep_rank_order_and_width():
    rows = [
        [entry(-0.1, 11), entry(-1.1, 91)],
        [entry(-0.2, 12), entry(-1.2, 92)],
        [entry(-0.3, 13), entry(-1.3, 93)],
    ]
    lp, ids = prompt_logprobs_from_meta(meta_topk(rows), 2, prompt_len=4)
    assert len(lp) == 4
    assert ids[0] == [11, 91] and lp[0] == [-0.1, -1.1]
    assert lp[-1] == [0.0, 0.0] and ids[-1] == [0, 0]


def test_a_short_row_raises_rather_than_being_padded():
    """0.0 is log(1) -- certainty -- so padding with it is not a neutral filler.

    An earlier version padded short rows to keep the tensor square. That put the
    largest value a logprob can take on a filler token, mid-sequence, inside the
    distillation KL, and produced a correctly shaped tensor that trains on
    nonsense. vLLM's own reader asserts the width; so does this.
    """
    rows = [[entry(-0.1, 11)], [entry(-0.2, 12), entry(-1.2, 92)], [entry(-0.3, 13)]]
    with pytest.raises(ValueError, match="certainty"):
        prompt_logprobs_from_meta(meta_topk(rows), 2, prompt_len=4)


def test_the_short_row_error_names_the_position():
    rows = [[entry(-0.1, 11), entry(-1.1, 91)], [entry(-0.2, 12)], [entry(-0.3, 13), entry(-1.3, 93)]]
    with pytest.raises(ValueError, match="position 1"):
        prompt_logprobs_from_meta(meta_topk(rows), 2, prompt_len=4)


@pytest.mark.parametrize("n_rows", [0, 1, 2, 5, 10])
def test_a_wrong_row_count_raises_instead_of_being_sliced(n_rows):
    """The failure mode this guards is silent, so it must be made loud."""
    rows = [entry(-0.1, 11)] * n_rows
    if n_rows in (3, 4):  # 4 is the legal placeholder case for prompt_len=4
        pytest.skip("legal row count")
    with pytest.raises(ValueError, match="input-logprob rows"):
        prompt_logprobs_from_meta(meta_flat(rows), 0, prompt_len=4)


def test_the_error_says_what_it_refuses_to_do():
    with pytest.raises(ValueError) as exc:
        prompt_logprobs_from_meta(meta_flat([entry(-0.1, 11)]), 0, prompt_len=4)
    assert "silently corrupts" in str(exc.value)


def test_empty_meta_is_an_error_not_an_empty_result():
    """Returning nothing here would make the teacher train on a zero KL."""
    with pytest.raises(ValueError):
        prompt_logprobs_from_meta({}, 0, prompt_len=4)


def test_width_matches_what_the_consumer_indexes():
    """teacher_manager reads (S, 1 or K); K comes from the topk setting."""
    rows = [[entry(-0.1, 11)] * 4 for _ in range(3)]
    lp, ids = prompt_logprobs_from_meta(meta_topk(rows), 4, prompt_len=4)
    assert {len(r) for r in lp} == {4}
    assert {len(r) for r in ids} == {4}
