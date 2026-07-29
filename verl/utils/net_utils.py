# Copyright 2023-2024 SGLang Team
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
# ==============================================================================
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
import ipaddress
import socket


def is_ipv4(ip_str: str) -> bool:
    """
    Check if the given string is an IPv4 address

    Args:
        ip_str: The IP address string to check

    Returns:
        bool: Returns True if it's an IPv4 address, False otherwise
    """
    try:
        ipaddress.IPv4Address(ip_str)
        return True
    except ipaddress.AddressValueError:
        return False


def is_ipv6(ip_str: str) -> bool:
    """
    Check if the given string is an IPv6 address

    Args:
        ip_str: The IP address string to check

    Returns:
        bool: Returns True if it's an IPv6 address, False otherwise
    """
    try:
        ipaddress.IPv6Address(ip_str)
        return True
    except ipaddress.AddressValueError:
        return False


def is_valid_ipv6_address(address: str) -> bool:
    try:
        ipaddress.IPv6Address(address)
        return True
    except ValueError:
        return False


def get_free_port(address: str, with_alive_sock: bool = False) -> tuple[int, socket.socket | None]:
    """Find a free port on the given address.

    By default the socket is closed internally, suitable for immediate use.
    Set with_alive_sock=True to keep the socket open as a port reservation,
    preventing other calls from getting the same port. The caller is
    responsible for closing the socket before the port is actually bound
    by the target service (e.g. NCCL, uvicorn).

    Note that with_alive_sock does NOT hold the port against a *wildcard* bind:
    the socket carries SO_REUSEADDR, and two SO_REUSEADDR sockets are allowed to
    hold the same port as long as one is bound to a specific address and the
    other to the wildcard. Use reserve_static_port when the reservation has to be
    exclusive.
    """
    family = socket.AF_INET6 if is_valid_ipv6_address(address) else socket.AF_INET

    sock = socket.socket(family=family, type=socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((address, 0))
    port = sock.getsockname()[1]
    if with_alive_sock:
        return port, sock
    sock.close()
    return port, None


# The kernel allocates bind(0) ports from ip_local_port_range, 32768-60999 by
# default. Anything picked from there and then released -- which is what a
# find-a-free-port helper does -- can be handed to somebody else before its
# intended owner gets around to binding it for real. Ports below the range are
# never handed out that way, so a slot down here is only ever taken by another
# process that also asked for it explicitly.
STATIC_PORT_FLOOR = 20000
STATIC_PORT_CEIL = 30000


def _kernel_reserved_ports() -> set[int]:
    """Ports the administrator excluded from ephemeral allocation.

    An explicit bind still succeeds on these, so whatever they were set aside for
    would be trampled rather than told the port is taken. They have to be skipped
    by hand.
    """
    try:
        with open("/proc/sys/net/ipv4/ip_local_reserved_ports") as fh:
            raw = fh.read().strip()
    except OSError:
        return set()

    reserved: set[int] = set()
    for part in (p.strip() for p in raw.split(",")):
        if not part:
            continue
        low, _, high = part.partition("-")
        try:
            reserved.update(range(int(low), int(high or low) + 1))
        except ValueError:
            continue
    return reserved


def reserve_static_port(slot: int, family: int = socket.AF_INET) -> tuple[int, socket.socket]:
    """Exclusively reserve a port outside the ephemeral range, starting at `slot`.

    Returns the port and the socket holding it. The socket binds the wildcard
    address and deliberately does NOT set SO_REUSEADDR, so the reservation is
    exclusive against every other bind including wildcard ones. The caller must
    close it immediately before the port is bound for real -- releasing it any
    earlier is the same race this exists to close.

    `slot` selects the starting port, so callers that can number themselves
    uniquely never contend with each other; the scan only has to step around
    unrelated processes on the host.
    """
    span = STATIC_PORT_CEIL - STATIC_PORT_FLOOR
    reserved = _kernel_reserved_ports()
    wildcard = "::" if family == socket.AF_INET6 else ""

    for offset in range(span):
        port = STATIC_PORT_FLOOR + (slot + offset) % span
        if port in reserved:
            continue
        sock = socket.socket(family=family, type=socket.SOCK_STREAM)
        try:
            sock.bind((wildcard, port))
        except OSError:
            sock.close()
            continue
        return port, sock

    raise RuntimeError(
        f"no free port in [{STATIC_PORT_FLOOR}, {STATIC_PORT_CEIL}) after scanning all {span}"
    )
