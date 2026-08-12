"""Test-side shim over the packaged conformance suite.

The reusable pieces live in ``moonep.alloc_contract`` so out-of-tree policies
(e.g. MLB) can import them from an installed wheel, where ``tests/`` is absent.
Only ``reference_alloc`` stays here: it drives MoonEP's torch planning
reference, which is test-only code.
"""

import torch

from moonep.alloc_contract import (  # noqa: F401  (re-exported for tests)
    GEOMETRIES,
    TPE_KINDS,
    TPE_KINDS_NEEDING_DIVISIBLE,
    AllocGeometry,
    alloc_invariant_errors,
    b_pressure,
    check_policy,
    make_tpe,
    max_rank_load,
)


def reference_alloc(tpe: torch.Tensor, geom: AllocGeometry) -> torch.Tensor:
    """Drive MoonEP's torch reference policy on a synthetic ``tpe``.

    Pure host code: no CUDA, no process group. ``topk_experts`` is required by
    the reference signature but unused on the ``return_alloc`` path.
    """
    from tests.planning_reference import launch_planning_torch_reference

    dummy_topk = torch.zeros(geom.S, geom.K, dtype=torch.int32)
    return launch_planning_torch_reference(
        geom.ctx(), dummy_topk, tpe, return_alloc=True
    )
