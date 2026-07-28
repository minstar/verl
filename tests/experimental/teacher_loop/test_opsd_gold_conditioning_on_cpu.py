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

"""CPU unit tests for OPSD gold-answer teacher conditioning helpers.

Covers: per-sample ground-truth extraction (reward_model.ground_truth with
extra_info.correct_answer fallback), gold-turn rendering via the chat template
(BOS stripping, {gold} substitution), and the injected-span stitching that
excludes the injected tokens from the returned teacher logprobs.
"""

import numpy as np
import pytest
import torch
from tensordict import TensorDict

import verl.experimental.teacher_loop.teacher_manager as tm
from verl.protocol import DataProto


def _data_proto_with(non_tensor: dict) -> DataProto:
    n = len(next(iter(non_tensor.values())))
    batch = TensorDict({"dummy": torch.zeros(n, 1)}, batch_size=[n])
    non_tensor_np = {k: np.array(v, dtype=object) for k, v in non_tensor.items()}
    return DataProto(batch=batch, non_tensor_batch=non_tensor_np)


class _StubTokenizer:
    """Minimal tokenizer stub: 1 token per whitespace-separated word."""

    bos_token = "<bos>"

    def __init__(self):
        self.vocab = {}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert len(messages) == 1 and messages[0]["role"] == "user"
        return f"<bos><user>{messages[0]['content']}</user>"

    def encode(self, text, add_special_tokens=False):
        tokens = text.split()
        return [self.vocab.setdefault(tok, len(self.vocab) + 1) for tok in tokens]


# ── _extract_gold_text ──────────────────────────────────────────────


def test_extract_gold_prefers_reward_model_ground_truth():
    data = _data_proto_with(
        {
            "reward_model": [{"ground_truth": "D"}, {"ground_truth": "yes"}],
            "extra_info": [{"correct_answer": "other"}, {"correct_answer": "other2"}],
        }
    )
    assert tm._extract_gold_text(data, 0) == "D"
    assert tm._extract_gold_text(data, 1) == "yes"


def test_extract_gold_falls_back_to_correct_answer_when_gt_empty():
    data = _data_proto_with(
        {
            "reward_model": [{"ground_truth": ""}, {"ground_truth": "   "}],
            "extra_info": [
                {"correct_answer": "Diagnosis: acute renal failure."},
                {"correct_answer": "Differential diagnosis of acute weakness."},
            ],
        }
    )
    assert tm._extract_gold_text(data, 0) == "Diagnosis: acute renal failure."
    assert tm._extract_gold_text(data, 1) == "Differential diagnosis of acute weakness."


def test_extract_gold_from_fields_directly():
    """Field-level helper used by the streaming (agent-loop) teacher path."""
    assert tm._extract_gold_text_from_fields({"ground_truth": "B"}, {"correct_answer": "x"}) == "B"
    assert tm._extract_gold_text_from_fields({"ground_truth": ""}, {"correct_answer": "sol"}) == "sol"
    assert tm._extract_gold_text_from_fields(None, None) is None
    assert tm._extract_gold_text_from_fields("not-a-dict", 3.14) is None


def test_extract_gold_returns_none_when_nothing_available():
    data = _data_proto_with(
        {
            "reward_model": [{"ground_truth": ""}],
            "extra_info": [{"correct_answer": ""}],
        }
    )
    assert tm._extract_gold_text(data, 0) is None
    no_keys = _data_proto_with({"other": [{"x": 1}]})
    assert tm._extract_gold_text(no_keys, 0) is None


# ── _render_user_turn_tokens / _build_gold_tokens ───────────────────


def test_render_user_turn_strips_bos_and_uses_template():
    tok = _StubTokenizer()
    ids = tm._render_user_turn_tokens("hello world", tok)
    # BOS must be stripped: the rendered string starts with "<user>hello",
    # so the first token is "<user>hello", never "<bos><user>hello".
    rendered_words = "<user>hello world</user>".split()
    assert ids == [tok.vocab[w] for w in rendered_words]
    assert "<bos><user>hello" not in tok.vocab


def test_build_gold_tokens_substitutes_and_caches(monkeypatch):
    tok = _StubTokenizer()
    tm._GOLD_TOKEN_CACHE.clear()
    manager = tm.AsyncTeacherLLMServerManager.__new__(tm.AsyncTeacherLLMServerManager)
    gold_with_braces = "Answer: {D}"  # braces must not break rendering (str.replace, not format)
    ids = manager._build_gold_tokens(gold_with_braces, tok)
    assert ids, "gold tokens must be produced"
    # The gold text must actually appear in the rendered turn.
    inv = {v: k for k, v in tok.vocab.items()}
    decoded = " ".join(inv[i] for i in ids)
    assert "{D}" in decoded
    assert "{gold}" not in decoded
    # Cached under the truncated gold text.
    assert tm._GOLD_TOKEN_CACHE[gold_with_braces] == ids
    # Second call hits the cache (same object).
    assert manager._build_gold_tokens(gold_with_braces, tok) is ids


def test_build_gold_tokens_requires_text_and_tokenizer():
    manager = tm.AsyncTeacherLLMServerManager.__new__(tm.AsyncTeacherLLMServerManager)
    assert manager._build_gold_tokens("", _StubTokenizer()) is None
    assert manager._build_gold_tokens("gold", None) is None


def test_build_gold_tokens_truncates_to_max_chars(monkeypatch):
    monkeypatch.setattr(tm, "_OPSD_GOLD_MAX_CHARS", 6)
    tm._GOLD_TOKEN_CACHE.clear()
    tok = _StubTokenizer()
    manager = tm.AsyncTeacherLLMServerManager.__new__(tm.AsyncTeacherLLMServerManager)
    manager._build_gold_tokens("abcdef_SHOULD_BE_DROPPED", tok)
    assert list(tm._GOLD_TOKEN_CACHE.keys()) == ["abcdef"]
    inv = {v: k for k, v in tok.vocab.items()}
    assert not any("SHOULD_BE_DROPPED" in w for w in inv.values())


# ── _stitch_out_injected_span ───────────────────────────────────────


def test_stitch_removes_injected_span_exactly():
    # original sequence: prompt of 3, response of 4 → target_len 7; inject 2 tokens at pos 3.
    orig_len, insert_pos, inject_len = 7, 3, 2
    full_logprobs = torch.arange(0, orig_len + inject_len, dtype=torch.float32)  # 0..8
    full_ids = torch.arange(100, 100 + orig_len + inject_len, dtype=torch.int32)
    ids, logprobs = tm._stitch_out_injected_span(
        full_ids, full_logprobs, insert_pos=insert_pos, inject_len=inject_len, target_len=orig_len
    )
    assert logprobs.tolist() == [0.0, 1.0, 2.0, 5.0, 6.0, 7.0, 8.0]
    assert ids.tolist() == [100, 101, 102, 105, 106, 107, 108]
    assert logprobs.shape[0] == ids.shape[0] == orig_len


def test_stitch_short_output_falls_back_and_pads():
    # Server returned fewer positions than expected: fall back to truncation + right-pad.
    target_len = 7
    full_logprobs = torch.arange(0, 5, dtype=torch.float32)
    full_ids = torch.arange(100, 105, dtype=torch.int32)
    ids, logprobs = tm._stitch_out_injected_span(
        full_ids, full_logprobs, insert_pos=3, inject_len=2, target_len=target_len
    )
    assert logprobs.shape[0] == ids.shape[0] == target_len
    assert logprobs.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 0.0]


def test_stitch_prepend_case_removes_leading_span():
    # insert_pos=0 corresponds to the prepend fallback in compute_teacher_logprobs_single.
    full_logprobs = torch.tensor([9.0, 9.0, 0.0, 1.0, 2.0])
    full_ids = torch.tensor([7, 7, 100, 101, 102], dtype=torch.int32)
    ids, logprobs = tm._stitch_out_injected_span(
        full_ids, full_logprobs, insert_pos=0, inject_len=2, target_len=3
    )
    assert logprobs.tolist() == [0.0, 1.0, 2.0]
    assert ids.tolist() == [100, 101, 102]
