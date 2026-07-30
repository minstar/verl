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
"""Two teacher questions that look like one and are not.

`stream_teacher_with_rollout` selects HOW teacher logprobs are obtained.
`_teacher_needs_wake` selects WHETHER the teacher engine is slept and woken
around each rollout. Collapsing them back into one flag reintroduces a hang that
produces no log line at all: with free_cache_engine=False, sleep() releases
nothing while wake_up() still issues flush_cache() to an engine that was never
slept, and the run stops inside _validate before a single sample is dispatched.

The gate itself is exercised as an expression; the call sites are checked
structurally, because the failure mode is someone tidying the two names back
together and no unit test noticing.
"""

import ast
import inspect
from pathlib import Path

import pytest

import verl.experimental.agent_loop.agent_loop as agent_loop


def gate(enable_resource_pool: bool, free_cache_engine: bool) -> bool:
    """The condition as the manager computes it."""
    return enable_resource_pool and free_cache_engine


@pytest.mark.parametrize(
    "pool,free_cache,expected",
    [
        (True, True, True),  # dedicated teacher that really releases memory
        (True, False, False),  # the configuration that hung
        (False, True, False),
        (False, False, False),
    ],
)
def test_wake_gate_requires_both(pool, free_cache, expected):
    assert gate(pool, free_cache) is expected


def _source() -> str:
    return Path(inspect.getfile(agent_loop)).read_text(encoding="utf-8")


def _generate_sequences_body() -> str:
    """The AgentLoopManager.generate_sequences source, by AST rather than by grep."""
    tree = ast.parse(_source())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "AgentLoopManager":
            for item in node.body:
                if isinstance(item, ast.AsyncFunctionDef) and item.name == "generate_sequences":
                    return ast.get_source_segment(_source(), item)
    raise AssertionError("AgentLoopManager.generate_sequences not found")


def test_wake_and_sleep_are_gated_on_the_wake_flag():
    body = _generate_sequences_body()
    assert "teacher_model_manager.wake_up()" in body
    assert "teacher_model_manager.sleep()" in body
    assert "_teacher_needs_wake" in body, (
        "the wake/sleep pair must be gated on _teacher_needs_wake; gating it on "
        "stream_teacher_with_rollout is what hung the TT-OPD arm"
    )
    assert "stream_teacher_with_rollout" not in body, (
        "stream_teacher_with_rollout selects the teacher logprob path, not whether "
        "the engine is woken; using it here re-couples two different questions"
    )


def test_both_halves_share_one_condition():
    """A wake without a matching sleep, or the reverse, is the original defect."""
    body = _generate_sequences_body()
    tree = ast.parse(body.strip())
    gated = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.If)
        and isinstance(n.test, ast.Attribute)
        and n.test.attr == "_teacher_needs_wake"
    ]
    assert len(gated) == 2, f"expected wake and sleep each gated once, found {len(gated)}"


def test_flag_is_defined_on_every_construction_path():
    """Both the distillation and non-distillation branches must set it."""
    src = _source()
    assert src.count("self._teacher_needs_wake = ") == 2, (
        "one branch of __init__ leaves _teacher_needs_wake undefined, which turns "
        "the hang into an AttributeError at the first rollout"
    )
