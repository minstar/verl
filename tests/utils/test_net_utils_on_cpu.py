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
"""Port allocation for colocated rollout servers.

These cover reserve_static_port, which exists because picking a port with bind(0)
and releasing it hands out a number nobody owns until the eventual binder gets
around to it. test_draw_and_release_collides demonstrates the failure the helper
is meant to remove; the rest pin the properties that removing it depends on.
"""

import socket
from types import SimpleNamespace

import pytest

from verl.utils.net_utils import (
    STATIC_PORT_CEIL,
    STATIC_PORT_FLOOR,
    _kernel_reserved_ports,
    get_free_port,
    reserve_static_port,
)


def _draw_and_release() -> int:
    """What SGLang's PortArgs.init_new does: bind(0), read the port, close."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def _ephemeral_range() -> tuple[int, int]:
    with open("/proc/sys/net/ipv4/ip_local_port_range") as fh:
        low, high = fh.read().split()
    return int(low), int(high)


def test_draw_and_release_collides():
    """The motivating race: concurrent draws are not distinct.

    A colocated run draws once per rollout replica, plus once per teacher replica
    under self-distillation, and each drawn port stays unbound for as long as it
    takes the scheduler subprocess to spawn and import torch. Any duplicate inside
    that window kills whichever server binds second.
    """
    rounds_with_a_duplicate = sum(1 for _ in range(2000) if len({_draw_and_release() for _ in range(8)}) < 8)
    assert rounds_with_a_duplicate > 0, (
        "expected draw-and-release to produce duplicates; if this ever stops being "
        "true the kernel's allocation policy changed and reserve_static_port's "
        "rationale should be re-checked rather than the test relaxed"
    )


def test_distinct_slots_never_collide():
    ports, held = [], []
    try:
        for slot in range(24):
            port, sock = reserve_static_port(slot)
            ports.append(port)
            held.append(sock)
        assert len(set(ports)) == len(ports)
    finally:
        for sock in held:
            sock.close()


def test_reservation_is_exclusive_against_wildcard():
    port, sock = reserve_static_port(0)
    try:
        other = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        other.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        with pytest.raises(OSError):
            other.bind(("", port))
        other.close()
    finally:
        sock.close()


def test_get_free_port_reservation_is_not_exclusive():
    """Documents why reserve_static_port exists alongside get_free_port.

    get_free_port sets SO_REUSEADDR, and two SO_REUSEADDR sockets may share a port
    when one is bound to a specific address and the other to the wildcard -- so its
    "reservation" does not stop the bind it is supposed to stop.
    """
    address = socket.gethostbyname(socket.gethostname())
    port, sock = get_free_port(address, with_alive_sock=True)
    try:
        other = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        other.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        other.bind(("", port))  # succeeds despite the live "reservation"
        other.close()
    finally:
        sock.close()


def test_ports_avoid_ephemeral_and_reserved_ranges():
    low, high = _ephemeral_range()
    reserved = _kernel_reserved_ports()
    held = []
    try:
        for slot in range(0, STATIC_PORT_CEIL - STATIC_PORT_FLOOR, 997):
            port, sock = reserve_static_port(slot)
            held.append(sock)
            assert not low <= port <= high
            assert port not in reserved
            assert STATIC_PORT_FLOOR <= port < STATIC_PORT_CEIL
    finally:
        for sock in held:
            sock.close()


def test_same_slot_scans_past_a_taken_port():
    """Two jobs on one host can pick the same slot; the scan has to step around."""
    first, first_sock = reserve_static_port(0)
    second, second_sock = reserve_static_port(0)
    try:
        assert first != second
    finally:
        first_sock.close()
        second_sock.close()


def test_released_port_is_bindable():
    port, sock = reserve_static_port(123)
    sock.close()
    rebound = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    rebound.bind(("", port))
    rebound.close()


def test_nccl_port_slot_separates_roles_and_ranks():
    pytest.importorskip("sglang")
    from verl.workers.rollout.sglang_rollout.async_sglang_server import (
        _ROLE_SLOT_SPAN,
        SGLangHttpServer,
    )

    def slot(name, replica_rank, node_rank=0, nnodes=1):
        server = SimpleNamespace(replica_rank=replica_rank, node_rank=node_rank, nnodes=nnodes)
        # get_actor_name() is unavailable outside an actor; the method falls back
        # to the policy role, so the role is injected the way ray would report it.
        server._actor_name = name
        import unittest.mock as mock

        with mock.patch(
            "ray.get_runtime_context",
            return_value=SimpleNamespace(get_actor_name=lambda: name),
        ):
            return SGLangHttpServer._nccl_port_slot(server)

    policy = [slot(f"sglang_server_{r}_0", r) for r in range(7)]
    teacher = [slot("sglang_server_teacher_0_0", 0)]
    reward = [slot("sglang_server_reward_0_0", 0)]

    assert len(set(policy)) == 7
    assert set(policy).isdisjoint(teacher)
    assert set(policy).isdisjoint(reward)
    assert set(teacher).isdisjoint(reward)
    assert teacher[0] == _ROLE_SLOT_SPAN
