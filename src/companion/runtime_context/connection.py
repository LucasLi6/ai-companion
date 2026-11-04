"""Trusted client-IP extraction from real connection metadata.

An IP address may only come from the transport layer. It is never inferred
from the message text, a model output, or a user-supplied field.

``X-Forwarded-For`` is only honoured when the *actual* peer address belongs to
an explicitly configured trusted proxy list. The rightmost address that is not
itself a trusted proxy is used, which is the standard safe interpretation
(everything left of it may have been appended by the client). Anything
private, loopback, link-local, multicast or otherwise non-global is rejected:
it cannot be geo-located and must never be looked up.

The raw address is used in memory only. It is not persisted and not logged;
:class:`GeoLocation` keeps only the resolved, city-level fields.
"""

from __future__ import annotations

import ipaddress
from dataclasses import dataclass
from typing import Iterable, Optional

# Maximum accepted length of an X-Forwarded-For chain. A hostile client can
# send an arbitrarily long header; beyond this it is ignored entirely.
MAX_FORWARDED_FOR_HOPS = 8


@dataclass(frozen=True)
class ConnectionInfo:
    """Connection metadata supplied by the transport layer.

    ``device_timezone`` and ``device_trusted`` describe an authenticated
    device. An untrusted device name is recorded but never used to select the
    display time zone.
    """

    remote_ip: Optional[str] = None
    forwarded_for: Optional[str] = None
    device_timezone: Optional[str] = None
    device_trusted: bool = False
    device_id: Optional[str] = None


def _normalise_ip(value: str | None) -> Optional[str]:
    """Parse an address (public or private) without judging its scope."""
    if not value:
        return None
    text = value.strip()
    if not text or len(text) > 64:
        return None
    # Strip a bracketed IPv6 form and an optional port.
    if text.startswith("["):
        end = text.find("]")
        if end != -1:
            text = text[1:end]
    elif text.count(":") == 1 and "." in text:
        text = text.split(":", 1)[0]
    try:
        address = ipaddress.ip_address(text)
    except ValueError:
        return None
    return str(address)


def _public_ip(value: str | None) -> Optional[str]:
    """Return a normalised public IP, or ``None`` for anything else."""
    text = _normalise_ip(value)
    if text is None:
        return None
    if not ipaddress.ip_address(text).is_global:
        return None
    return text


def _trusted_set(trusted_proxies: Iterable[str] | None) -> set[str]:
    trusted: set[str] = set()
    for item in trusted_proxies or ():
        try:
            network = ipaddress.ip_network(str(item).strip(), strict=False)
        except ValueError:
            continue
        trusted.add(str(network))
    return trusted


def _is_trusted(address: str, trusted: set[str]) -> bool:
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return False
    return any(ip in ipaddress.ip_network(net) for net in trusted)


def resolve_client_ip(
    connection: ConnectionInfo | None,
    *,
    trusted_proxies: Iterable[str] | None = None,
) -> Optional[str]:
    """Return the public client IP, or ``None`` when it cannot be trusted.

    A private/loopback peer is never returned as a locatable client, but it
    *may* be a trusted reverse proxy: XFF is then honoured only if the peer is
    explicitly listed. A private hop inside the chain yields ``None`` (nothing
    to geolocate), never a fallback to the proxy's own address.
    """
    if connection is None:
        return None
    peer = _normalise_ip(connection.remote_ip)
    if peer is None:
        return None

    trusted = _trusted_set(trusted_proxies)
    forwarded = (connection.forwarded_for or "").strip()
    if not forwarded or not _is_trusted(peer, trusted):
        return _public_ip(peer)

    hops = [part.strip() for part in forwarded.split(",") if part.strip()]
    if not hops or len(hops) > MAX_FORWARDED_FOR_HOPS:
        return _public_ip(peer)
    for hop in reversed(hops):
        candidate = _normalise_ip(hop)
        if candidate is None:
            # Malformed hop: the chain cannot be trusted for location.
            return None
        if _is_trusted(candidate, trusted):
            continue
        # The rightmost untrusted hop is the client. If it is private or
        # reserved, there is nothing legitimate to look up.
        return _public_ip(candidate)
    # Every hop was a trusted proxy; no client address remains.
    return None
