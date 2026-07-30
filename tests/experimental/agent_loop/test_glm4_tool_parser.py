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
"""GLM-4 emits tool calls as `name\\n{json}` and carries no <tool_call> marker.

test_hermes_parses_nothing_from_glm_output is the regression that motivated the
parser: under hermes the calls are invisible, the turn yields no observation, and
the trajectory collapses.
"""

import asyncio
import json

import pytest

from verl.experimental.agent_loop.tool_parser import ToolParser


class _Fn:
    def __init__(self, name):
        self.name = name


class _Tool:
    def __init__(self, name):
        self.function = _Fn(name)


TOOLS = [_Tool("search_pubmed"), _Tool("search_evidence")]


class _Tokenizer:
    """Decodes the list of ints back to the text it was built from.

    The parsers only ever call decode, so a real tokenizer is unnecessary here and
    would tie the test to a checkpoint on disk.
    """

    def __init__(self):
        self._table = {}

    def encode(self, text):
        ids = [len(self._table)]
        self._table[ids[0]] = text
        return ids

    def decode(self, ids):
        return "".join(self._table[i] for i in ids)


def _extract(parser_name, text, tools=TOOLS):
    tok = _Tokenizer()
    parser = ToolParser.get_tool_parser(parser_name, tok)
    content, calls = asyncio.run(parser.extract_tool_calls(tok.encode(text), tools))
    return content, [(c.name, json.loads(c.arguments)) for c in calls]


def test_parses_the_shape_glm4_actually_emits():
    _, calls = _extract("glm4", 'search_pubmed\n{"query": "lisinopril spironolactone interaction"}')
    assert calls == [("search_pubmed", {"query": "lisinopril spironolactone interaction"})]


def test_reasoning_before_the_call_is_kept_as_content():
    content, calls = _extract("glm4", 'I should look this up.\nsearch_evidence\n{"query": "sepsis bundle"}')
    assert calls == [("search_evidence", {"query": "sepsis bundle"})]
    assert "I should look this up." in content
    assert "search_evidence" not in content


def test_multi_line_json_arguments():
    text = 'search_evidence\n{\n  "query": "sepsis",\n  "max_results": 2\n}'
    _, calls = _extract("glm4", text)
    assert calls == [("search_evidence", {"query": "sepsis", "max_results": 2})]


def test_a_name_that_is_not_a_registered_tool_is_not_executed():
    """Without this an assistant turn that merely discusses JSON becomes a call."""
    _, calls = _extract("glm4", 'rm_rf\n{"path": "/"}')
    assert calls == []


def test_prose_followed_by_json_is_not_a_call():
    _, calls = _extract("glm4", 'answer\n{"a": 1}')
    assert calls == []


def test_plain_answer_yields_no_calls_and_survives_intact():
    content, calls = _extract("glm4", "The answer is A.")
    assert calls == []
    assert content == "The answer is A."


def test_malformed_arguments_are_dropped_not_raised():
    _, calls = _extract("glm4", 'search_pubmed\n{"query": ')
    assert calls == []


def test_non_object_arguments_are_rejected():
    _, calls = _extract("glm4", "search_pubmed\n[1, 2, 3]")
    assert calls == []


def test_hermes_parses_nothing_from_glm_output():
    """The defect this parser exists to fix, pinned so it cannot be re-introduced."""
    _, calls = _extract("hermes", 'search_pubmed\n{"query": "x"}')
    assert calls == []


@pytest.mark.parametrize("parser", ["glm4", "hermes"])
def test_empty_response_yields_no_calls(parser):
    _, calls = _extract(parser, "")
    assert calls == []
