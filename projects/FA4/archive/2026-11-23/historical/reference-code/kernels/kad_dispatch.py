"""Knowledge-aware dispatch (KAD) for qualified FA4 configurations.

KAD deliberately stays outside the pinned FA4 source tree.  It consumes a small
JSON policy containing measured lower bounds and chooses a configuration only
when the evidence is strong enough for the requested deployment threshold.
Unknown shapes and weak evidence return ``None`` so callers can use native FA4.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class KADDecision:
    shape_id: str
    config: Mapping[str, Any]
    lower_bound: float
    source: str


def _key(shape: Mapping[str, Any]) -> str:
    # Stable identity shared by benchmark matrix and launcher readbacks.
    return json.dumps(
        [
            [shape["b"], shape["sq"], shape["hq"], shape["d"]],
            [shape["b"], shape["sk"], shape["hk"], shape["d"]],
            [shape["b"], shape["sk"], shape["hk"], shape["d"]],
            shape["dtype"],
            bool(shape["causal"]),
        ],
        separators=(",", ":"),
    )


def load_policy(path: str | Path) -> dict[str, Any]:
    policy = json.loads(Path(path).read_text())
    if policy.get("schema_version") != 1:
        raise ValueError("unsupported KAD policy schema")
    threshold = policy.get("minimum_lower_bound")
    if not isinstance(threshold, (int, float)) or threshold <= 1:
        raise ValueError("minimum_lower_bound must be greater than one")
    return policy


def choose(policy: Mapping[str, Any], shape: Mapping[str, Any], *, source: str = "KAD") -> KADDecision | None:
    """Return a qualified decision, or ``None`` for native fallback.

    ``lower_bound`` is a ratio (1.10 means a 10% speedup).  A candidate must
    also declare zero measured regressions and an explicit correctness gate.
    """
    row = policy.get("shapes", {}).get(_key(shape))
    # PR-wide acceptance uses case shapes that do not share the small release
    # matrix's exact B/S identity.  Pattern rows are opt-in policy entries and
    # still require the same correctness/regression/lower-bound gates.  We do
    # not synthesize patterns from timings at runtime.
    if row is None:
        for candidate in policy.get("patterns", []):
            match = candidate.get("match", {})
            if all(shape.get(k) == v for k, v in match.items()):
                row = candidate
                break
    if not row:
        return None
    if not row.get("correctness_pass", False) or row.get("regressions", 1) != 0:
        return None
    lower = float(row.get("lower_bound", 0.0))
    if lower < float(policy["minimum_lower_bound"]):
        return None
    config = row.get("config")
    if not isinstance(config, dict):
        return None
    return KADDecision(str(row.get("id", "unknown")), config, lower, source)
