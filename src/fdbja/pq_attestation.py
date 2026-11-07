# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Quantova Inc

"""Optional post quantum attestation of evaluation results.

Signs a canonical results manifest, such as a system name with its metric
scores, using FIPS 204 ML-DSA so a reader can confirm the reported scores are
authentic and have not been altered, with a signature scheme that stays safe
once large quantum computers exist.

The attestation is self contained. It carries the public key and the signature,
so a verifier needs only the attestation and the results. Integrity is always
covered by the payload digest and authenticity is covered when the verifier
pins the expected public key.

The scheme is provided by the optional pq extra. Install it with
``pip install fdbja[pq]``.
"""

import base64
import hashlib
import json
from typing import Any

SCHEME = "ML-DSA-65"
ISSUER = "Quantova"


def _backend() -> Any:
    """Return the ML-DSA backend or raise if the pq extra is not installed."""
    try:
        from dilithium_py.ml_dsa import ML_DSA_65
    except ImportError as error:
        msg = "Post quantum attestation needs the pq extra. Install it with pip install fdbja[pq]."
        raise RuntimeError(msg) from error
    return ML_DSA_65


def canonicalize(results: dict[str, Any]) -> bytes:
    """Return the canonical byte encoding an attestation is computed over."""
    return json.dumps(results, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def generate_keypair() -> tuple[bytes, bytes]:
    """Return a fresh (public_key, secret_key) pair for the signer."""
    public_key, secret_key = _backend().keygen()
    return public_key, secret_key


def build_attestation(payload: bytes, public_key: bytes, secret_key: bytes) -> dict[str, str]:
    """Return a self contained attestation over payload."""
    signature = _backend().sign(secret_key, payload, deterministic=True)
    return {
        "scheme": SCHEME,
        "issuer": ISSUER,
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "public_key": base64.b64encode(public_key).decode("ascii"),
        "signature": base64.b64encode(signature).decode("ascii"),
    }


def verify_attestation(
    payload: bytes, attestation: dict[str, str], expected_public_key: bytes | None = None
) -> bool:
    """Return True when the attestation matches payload.

    The payload digest is always checked. When expected_public_key is given the
    signer is pinned too, so a valid signature made under a different key is
    rejected.
    """
    if attestation.get("payload_sha256") != hashlib.sha256(payload).hexdigest():
        return False
    try:
        public_key = base64.b64decode(attestation["public_key"], validate=True)
        signature = base64.b64decode(attestation["signature"], validate=True)
    except (ValueError, KeyError):
        return False
    if expected_public_key is not None and public_key != expected_public_key:
        return False
    try:
        return bool(_backend().verify(public_key, payload, signature))
    except (ValueError, TypeError):
        return False


def save_signer(path: str, public_key: bytes, secret_key: bytes) -> None:
    """Write a signer key file that load_signer reads back."""
    data = {
        "scheme": SCHEME,
        "public_key": base64.b64encode(public_key).decode("ascii"),
        "secret_key": base64.b64encode(secret_key).decode("ascii"),
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle)


def load_signer(path: str) -> tuple[bytes, bytes]:
    """Read (public_key, secret_key) from a signer key file."""
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    return base64.b64decode(data["public_key"]), base64.b64decode(data["secret_key"])
