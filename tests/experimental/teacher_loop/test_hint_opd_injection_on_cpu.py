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

"""CPU unit tests for TT-OPD outcome-conditioned privileged hint injection.

These cover the defect that made hints a no-op for the entire v15-v32 run series:
the streaming teacher path never forwarded hint tokens, and no construction site
passed a tokenizer to the hint builder. Each test pins one link of the chain.

  * a hint is built and FORWARDED through the agent-loop streaming path
  * correct and incorrect trajectories get DIFFERENT hint token spans
  * the injected span is stitched back out, so the returned teacher logprobs stay
    aligned with the (uninjected) student sequence
  * multimodal samples are skipped
  * a missing / NaN / non-numeric score suppresses injection instead of guessing
  * gold conditioning takes precedence over hints when both are enabled
  * the manager refuses to be constructed without a tokenizer while conditioning is on
"""

import asyncio

import pytest
import torch

import verl.experimental.teacher_loop.teacher_manager as tm
from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput, AgentLoopWorker


class _StubTokenizer:
    """Minimal tokenizer: one token per whitespace-separated word, stable ids."""

    bos_token = "<bos>"

    def __init__(self):
        self.vocab: dict[str, int] = {}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert len(messages) == 1 and messages[0]["role"] == "user"
        return f"<bos><user> {messages[0]['content']} </user>"

    def encode(self, text, add_special_tokens=False):
        return [self.vocab.setdefault(tok, len(self.vocab) + 1) for tok in text.split()]


class _FakeTeacherOutput:
    def __init__(self, ids, logprobs):
        self.extra_fields = {"prompt_ids": ids, "prompt_logprobs": logprobs}


class _StubManager(tm.AsyncTeacherLLMServerManager):
    """Manager with the ray plumbing removed; the teacher returns position indices
    as logprobs so stitching can be checked exactly."""

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        self.pad_token_id = 0
        self.distillation_config = None
        self.distillation_loss_config = None
        self.calls: list[list[int]] = []

    @property
    def captured_prompt_ids(self):
        return self.calls[-1] if self.calls else None

    async def generate(self, request_id, prompt_ids, sampling_params, image_data=None, video_data=None):
        self.calls.append(list(prompt_ids))
        return _FakeTeacherOutput(list(prompt_ids), [float(i) for i in range(len(prompt_ids))])


@pytest.fixture
def hint_mode(monkeypatch):
    """HINT_OPD_ENABLED=1, gold off, caches cleared, known hint texts."""
    monkeypatch.setattr(tm, "_HINT_OPD_ENABLED", True)
    monkeypatch.setattr(tm, "_OPSD_GOLD_CONDITIONING", False)
    monkeypatch.setattr(tm, "_HINT_CORRECT", "Hint the answer is correct")
    monkeypatch.setattr(tm, "_HINT_INCORRECT", "Hint the answer is wrong reconsider it")
    monkeypatch.setattr(tm, "_HINT_TOKEN_CACHE", {})
    monkeypatch.setattr(tm, "_get_teacher_sampling_params", lambda a, b: {})
    return tm


@pytest.fixture
def manager(hint_mode):
    return _StubManager(_StubTokenizer())


def _worker(manager) -> AgentLoopWorker:
    """AgentLoopWorker with only the attributes _compute_teacher_logprobs touches.

    __new__ rather than __init__: the real constructor wants ray server handles and
    a model checkpoint on disk.
    """
    worker = AgentLoopWorker.__new__(AgentLoopWorker)
    worker.stream_teacher_with_rollout = True
    worker.tokenizer = manager._tokenizer
    worker.teacher_server_manager = manager
    worker._injection_reasons = {}
    worker._injection_seen = 0
    worker._injection_log_every = 0
    return worker


def _output(prompt_ids, response_ids, reward_score, multi_modal_data=None) -> AgentLoopOutput:
    return AgentLoopOutput(
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        response_mask=[1] * len(response_ids),
        reward_score=reward_score,
        multi_modal_data=multi_modal_data,
        metrics=AgentLoopMetrics(),
    )


def _run_stream(worker, output) -> None:
    asyncio.run(
        worker._compute_teacher_logprobs(
            output,
            prompt_ids=output.prompt_ids,
            response_ids=output.response_ids,
            validate=False,
            sample_kwargs={},
        )
    )


# ── hints are built and forwarded through the streaming path ────────────────


def test_streaming_path_forwards_hint_tokens(manager):
    """The v26-v32 defect: this call used to omit hint_token_ids entirely."""
    worker = _worker(manager)
    prompt_ids, response_ids = [10, 11, 12], [20, 21, 22]
    output = _output(prompt_ids, response_ids, reward_score=1.1)

    _run_stream(worker, output)

    hint_ids = manager._build_hint_tokens(True, manager._tokenizer)
    assert hint_ids, "hint tokens should have been built"
    # The teacher was asked to score prompt + HINT + response, not prompt + response.
    assert manager.captured_prompt_ids == prompt_ids + hint_ids + response_ids
    assert worker._injection_reasons == {tm.INJECT_HINT_CORRECT: 1}


def test_streaming_path_without_hints_scores_the_bare_sequence(manager, monkeypatch):
    monkeypatch.setattr(tm, "_HINT_OPD_ENABLED", False)
    worker = _worker(manager)
    output = _output([10, 11, 12], [20, 21], reward_score=1.1)

    _run_stream(worker, output)

    assert manager.captured_prompt_ids == [10, 11, 12, 20, 21]
    assert worker._injection_reasons == {tm.SKIP_DISABLED: 1}


# ── correct vs incorrect must differ ────────────────────────────────────────


def test_correct_and_incorrect_hints_differ(manager):
    correct, r_correct = tm.resolve_privileged_injection(
        manager, manager._tokenizer, has_multimodal=False, reward_score=1.1
    )
    incorrect, r_incorrect = tm.resolve_privileged_injection(
        manager, manager._tokenizer, has_multimodal=False, reward_score=0.0
    )

    assert (r_correct, r_incorrect) == (tm.INJECT_HINT_CORRECT, tm.INJECT_HINT_INCORRECT)
    assert correct and incorrect
    assert correct != incorrect, "outcome conditioning is vacuous if both outcomes get the same hint"


def test_correctness_threshold_is_the_documented_one(manager):
    # hcgym cosine reward: correct >= 0.7, wrong in [-0.5, 0.0]. 0.0 is NOT correct.
    assert tm.hint_correctness(1.1) is True
    assert tm.hint_correctness(0.7) is True
    assert tm.hint_correctness(0.0) is False
    assert tm.hint_correctness(-0.3) is False
    assert tm.hint_correctness(-999.0) is False  # degenerate-response sentinel


def test_hint_tokens_are_cached_per_outcome(manager):
    first = manager._build_hint_tokens(True, manager._tokenizer)
    second = manager._build_hint_tokens(True, manager._tokenizer)
    assert first is second
    assert set(tm._HINT_TOKEN_CACHE) == {True}


# ── the injected span is stitched back out ──────────────────────────────────


def test_injected_span_is_stitched_out_and_lengths_align(manager):
    """Returned logprobs must line up with the student sequence, hint excluded."""
    sequence_ids = [10, 11, 12, 20, 21, 22]
    hint_ids = [90, 91]
    teacher_ids, teacher_logprobs = asyncio.run(
        manager.compute_teacher_logprobs_single(
            sequence_ids=sequence_ids,
            hint_token_ids=hint_ids,
            original_length=3,
        )
    )

    assert manager.captured_prompt_ids == [10, 11, 12, 90, 91, 20, 21, 22]
    assert teacher_ids.shape[0] == teacher_logprobs.shape[0] == len(sequence_ids)
    assert teacher_ids.tolist() == sequence_ids, "hint tokens must not survive into the aligned output"
    # The stub's logprob at each position is its index in the ENHANCED sequence, so
    # the response positions must carry 5,6,7 — i.e. conditioned on the hint.
    assert teacher_logprobs.tolist() == [0.0, 1.0, 2.0, 5.0, 6.0, 7.0]


def test_end_to_end_stream_output_has_student_length(manager):
    worker = _worker(manager)
    prompt_ids, response_ids = [10, 11, 12], [20, 21, 22, 23]
    output = _output(prompt_ids, response_ids, reward_score=0.0)

    _run_stream(worker, output)

    n = len(prompt_ids) + len(response_ids)
    assert output.extra_fields["teacher_ids"].shape[0] == n
    assert output.extra_fields["teacher_logprobs"].shape[0] == n
    assert output.extra_fields["teacher_ids"].tolist() == prompt_ids + response_ids


def test_stitch_helper_pads_a_short_teacher_output():
    ids, logprobs = tm._stitch_out_injected_span(
        torch.tensor([1, 2, 3], dtype=torch.int32),
        torch.tensor([0.0, 1.0, 2.0]),
        insert_pos=2,
        inject_len=4,
        target_len=6,
    )
    assert ids.shape[0] == logprobs.shape[0] == 6


# ── multimodal is skipped ───────────────────────────────────────────────────


def test_multimodal_sample_is_skipped(manager):
    worker = _worker(manager)
    prompt_ids, response_ids = [10, 11], [20, 21]
    output = _output(prompt_ids, response_ids, reward_score=1.1, multi_modal_data={"images": ["<img>"]})

    _run_stream(worker, output)

    assert manager.captured_prompt_ids == prompt_ids + response_ids
    assert worker._injection_reasons == {tm.SKIP_MULTIMODAL: 1}


def test_multimodal_video_sample_is_skipped(manager):
    ids, reason = tm.resolve_privileged_injection(
        manager, manager._tokenizer, has_multimodal=True, reward_score=1.1
    )
    assert (ids, reason) == (None, tm.SKIP_MULTIMODAL)


# ── a missing score suppresses injection ────────────────────────────────────


@pytest.mark.parametrize("score", [None, float("nan"), "not-a-number"])
def test_unusable_score_suppresses_injection(manager, score):
    """Never guess: an unconditioned teacher beats a teacher told the wrong outcome."""
    assert tm.hint_correctness(score) is None
    ids, reason = tm.resolve_privileged_injection(
        manager, manager._tokenizer, has_multimodal=False, reward_score=score
    )
    assert ids is None
    assert reason == tm.SKIP_SCORE_MISSING


def test_missing_score_on_streaming_path_scores_bare_sequence(manager):
    worker = _worker(manager)
    prompt_ids, response_ids = [10, 11], [20, 21]
    output = _output(prompt_ids, response_ids, reward_score=None)

    _run_stream(worker, output)

    assert manager.captured_prompt_ids == prompt_ids + response_ids
    assert worker._injection_reasons == {tm.SKIP_SCORE_MISSING: 1}


# ── gold takes precedence over hints ────────────────────────────────────────


def test_gold_takes_precedence_over_hints(manager, monkeypatch):
    monkeypatch.setattr(tm, "_OPSD_GOLD_CONDITIONING", True)
    monkeypatch.setattr(tm, "_GOLD_TOKEN_CACHE", {})

    ids, reason = tm.resolve_privileged_injection(
        manager,
        manager._tokenizer,
        has_multimodal=False,
        gold_text="the answer is C",
        reward_score=1.1,
    )

    assert reason == tm.INJECT_GOLD
    assert ids == manager._build_gold_tokens("the answer is C", manager._tokenizer)
    assert ids != manager._build_hint_tokens(True, manager._tokenizer)


def test_gold_enabled_but_missing_does_not_fall_back_to_hints(manager, monkeypatch):
    """Mixing two privileged signals across one run's samples is uninterpretable."""
    monkeypatch.setattr(tm, "_OPSD_GOLD_CONDITIONING", True)
    monkeypatch.setattr(tm, "_GOLD_TOKEN_CACHE", {})

    ids, reason = tm.resolve_privileged_injection(
        manager, manager._tokenizer, has_multimodal=False, gold_text=None, reward_score=1.1
    )

    assert ids is None
    assert reason == tm.SKIP_GOLD_MISSING


def test_hints_fire_when_only_hints_are_enabled(manager, monkeypatch):
    monkeypatch.setattr(tm, "_OPSD_GOLD_CONDITIONING", False)
    _, reason = tm.resolve_privileged_injection(
        manager, manager._tokenizer, has_multimodal=False, gold_text="the answer is C", reward_score=1.1
    )
    assert reason == tm.INJECT_HINT_CORRECT


def test_both_disabled_is_reported_as_disabled(manager, monkeypatch):
    monkeypatch.setattr(tm, "_HINT_OPD_ENABLED", False)
    monkeypatch.setattr(tm, "_OPSD_GOLD_CONDITIONING", False)
    ids, reason = tm.resolve_privileged_injection(
        manager, manager._tokenizer, has_multimodal=False, reward_score=1.1
    )
    assert (ids, reason) == (None, tm.SKIP_DISABLED)


# ── the tokenizer gate cannot silently re-break ─────────────────────────────


def test_tokenizer_parameter_has_no_default():
    import inspect

    param = inspect.signature(tm.AsyncTeacherLLMServerManager.__init__).parameters["tokenizer"]
    assert param.default is inspect.Parameter.empty, (
        "a default here is what let both construction sites drop the tokenizer for 18 run versions"
    )


def _construct_without_tokenizer(monkeypatch):
    from verl.workers.config import DistillationConfig

    # Skip only the ray server plumbing in the base class; the guard under test is
    # the real one in AsyncTeacherLLMServerManager.__init__.
    monkeypatch.setattr(tm.AsyncLLMServerManager, "__init__", lambda self, **kwargs: None)
    return tm.AsyncTeacherLLMServerManager(
        config=None,
        servers=[],
        load_balancer_handle=None,
        distillation_config=DistillationConfig(),
        pad_token_id=0,
        tokenizer=None,
    )


def test_construction_without_tokenizer_raises_while_conditioning_is_on(monkeypatch):
    monkeypatch.setattr(tm, "_HINT_OPD_ENABLED", True)
    monkeypatch.setattr(tm, "_OPSD_GOLD_CONDITIONING", False)
    with pytest.raises(ValueError, match="tokenizer=None"):
        _construct_without_tokenizer(monkeypatch)


def test_construction_without_tokenizer_is_only_a_warning_when_conditioning_is_off(monkeypatch, capsys):
    monkeypatch.setattr(tm, "_HINT_OPD_ENABLED", False)
    monkeypatch.setattr(tm, "_OPSD_GOLD_CONDITIONING", False)
    manager = _construct_without_tokenizer(monkeypatch)
    assert manager._tokenizer is None
    assert "tokenizer=None" in capsys.readouterr().out


def test_no_tokenizer_reports_a_distinct_skip_reason(manager):
    ids, reason = tm.resolve_privileged_injection(manager, None, has_multimodal=False, reward_score=1.1)
    assert (ids, reason) == (None, tm.SKIP_NO_TOKENIZER)


# ── the batch/colocate path shares the same resolver ────────────────────────


def test_batch_path_injects_per_sample_hints(manager):
    """v15-v25 took this path. It must now pick the hint per sample's own reward."""
    from tensordict import TensorDict

    from verl.protocol import DataProto

    prompt_width, response_width = 4, 3
    # sample 0: prompt [1,2], response [3,4]  (reward 1.0 -> correct)
    # sample 1: prompt [5],   response [6,7,8] (reward 0.0 -> incorrect)
    prompts = torch.tensor([[0, 0, 1, 2], [0, 0, 0, 5]])
    responses = torch.tensor([[3, 4, 0], [6, 7, 8]])
    attention_mask = torch.tensor([[0, 0, 1, 1, 1, 1, 0], [0, 0, 0, 1, 1, 1, 1]])
    rm_scores = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    data = DataProto(
        batch=TensorDict(
            {
                "prompts": prompts,
                "responses": responses,
                "input_ids": torch.cat([prompts, responses], dim=1),
                "attention_mask": attention_mask,
                "rm_scores": rm_scores,
            },
            batch_size=2,
        )
    )

    asyncio.run(manager.compute_teacher_logprobs_batch(data))

    correct = manager._build_hint_tokens(True, manager._tokenizer)
    incorrect = manager._build_hint_tokens(False, manager._tokenizer)
    assert manager.calls == [[1, 2] + correct + [3, 4], [5] + incorrect + [6, 7, 8]]


def test_batch_path_without_rm_scores_suppresses_injection(manager):
    from tensordict import TensorDict

    from verl.protocol import DataProto

    prompts = torch.tensor([[0, 0, 1, 2]])
    responses = torch.tensor([[3, 4, 0]])
    data = DataProto(
        batch=TensorDict(
            {
                "prompts": prompts,
                "responses": responses,
                "input_ids": torch.cat([prompts, responses], dim=1),
                "attention_mask": torch.tensor([[0, 0, 1, 1, 1, 1, 0]]),
            },
            batch_size=1,
        )
    )

    asyncio.run(manager.compute_teacher_logprobs_batch(data))

    assert manager.calls == [[1, 2, 3, 4]]


# ── the injection log line is greppable ─────────────────────────────────────


def test_injection_reason_log_line_leads_with_the_injected_count():
    line = tm.format_injection_reasons(
        {tm.INJECT_HINT_CORRECT: 3, tm.INJECT_HINT_INCORRECT: 4, tm.SKIP_SCORE_MISSING: 2}
    )
    assert line.startswith("injected=7 ")
    assert "skipped=2" in line
    assert "skip:score-missing=2" in line
