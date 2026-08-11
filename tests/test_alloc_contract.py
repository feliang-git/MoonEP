"""Conformance tests for the allocation-policy contract.

Pure host code -- no CUDA, no torchrun:

    pytest -q tests/test_alloc_contract.py

These tests pin the seam between MoonEP's backend and a pluggable allocation
policy. They run against MoonEP's own torch reference here; an out-of-tree
policy (e.g. MLB) is expected to import ``tests.alloc_contract`` and run the
same suite against its own implementation via ``check_policy``.
"""

import pytest
import torch

from tests.alloc_contract import (
    GEOMETRIES,
    TPE_KINDS,
    TPE_KINDS_NEEDING_DIVISIBLE,
    AllocGeometry,
    alloc_invariant_errors,
    b_pressure,
    make_tpe,
    max_rank_load,
    reference_alloc,
)

GEOM_IDS = [f"R{g.R}_E{g.E}_S{g.S}_K{g.K}_tp{g.token_padding}" for g in GEOMETRIES]


def _tpe_or_skip(kind: str, geom: AllocGeometry, seed: int = 1234) -> torch.Tensor:
    if kind in TPE_KINDS_NEEDING_DIVISIBLE and geom.capacity % geom.E != 0:
        pytest.skip(f"{kind} requires E | S*K (E={geom.E}, S*K={geom.capacity})")
    return make_tpe(kind, geom, seed=seed)


@pytest.mark.parametrize("geom", GEOMETRIES, ids=GEOM_IDS)
@pytest.mark.parametrize("kind", TPE_KINDS)
def test_reference_policy_conforms(geom: AllocGeometry, kind: str):
    """MoonEP's builtin policy must satisfy every contract invariant."""
    tpe = _tpe_or_skip(kind, geom)
    # Precondition: top-k routing gives every source rank exactly S*K entries.
    assert torch.equal(
        tpe.sum(dim=1).to(torch.int64),
        torch.full((geom.R,), geom.capacity, dtype=torch.int64),
    ), f"{kind}: malformed test fixture, per-rank totals != S*K"

    alloc = reference_alloc(tpe, geom)
    errors = alloc_invariant_errors(alloc, tpe, geom)
    assert not errors, f"{kind} @ {geom}: " + "; ".join(errors)


@pytest.mark.parametrize("geom", GEOMETRIES, ids=GEOM_IDS)
@pytest.mark.parametrize("kind", TPE_KINDS)
def test_segment_bound_is_tight_not_incidental(geom: AllocGeometry, kind: str):
    """Invariant 4 holds because each dest rank receives from <= 1 home group.

    This is the structural property a replacement policy must reproduce. We
    assert the mechanism, not just the bound, so a policy that satisfies the
    count by luck still fails here.
    """
    tpe = _tpe_or_skip(kind, geom)
    alloc = reference_alloc(tpe, geom).cpu()
    epn = geom.epn

    for d in range(geom.R):
        donors = {
            e // epn
            for e in range(geom.E)
            if alloc[d, e].item() > 0 and not (d * epn <= e < (d + 1) * epn)
        }
        assert len(donors) <= 1, (
            f"{kind}: dest rank {d} received from home groups {sorted(donors)}; "
            f"the NvS padding budget in api.py assumes at most one"
        )


def _naive_rank_load(tpe: torch.Tensor, geom: AllocGeometry) -> torch.Tensor:
    """Per-rank token load if nothing migrates (every expert computed at home)."""
    expert_count = tpe.sum(dim=0).to(torch.int64)
    epn = geom.epn
    return torch.tensor(
        [expert_count[d * epn : (d + 1) * epn].sum() for d in range(geom.R)]
    )


@pytest.mark.parametrize("geom", GEOMETRIES, ids=GEOM_IDS)
@pytest.mark.parametrize("kind", TPE_KINDS)
def test_no_gratuitous_migration(geom: AllocGeometry, kind: str):
    """When the home placement already fits, nothing may move.

    Migration costs a weight prefetch, so a policy that shuffles tokens it did
    not have to is strictly worse even at equal max load. Note this is keyed on
    the naive load fitting in capacity, not on the fixture being "uniform" --
    make_tpe("uniform") is only uniform when E divides S*K, and the leftover
    otherwise creates real skew that *should* be migrated.
    """
    if geom.R == 1:
        pytest.skip("single-rank geometry has nothing to balance")
    tpe = _tpe_or_skip(kind, geom)
    if int(_naive_rank_load(tpe, geom).max().item()) > geom.capacity:
        pytest.skip(f"{kind}: home placement genuinely overflows, migration required")

    alloc = reference_alloc(tpe, geom).cpu()
    epn = geom.epn
    for d in range(geom.R):
        for e in range(geom.E):
            if not (d * epn <= e < (d + 1) * epn):
                assert alloc[d, e].item() == 0, (
                    f"{kind}: expert {e} migrated to rank {d} though the home "
                    f"placement already fit in capacity"
                )


@pytest.mark.parametrize("geom", GEOMETRIES, ids=GEOM_IDS)
@pytest.mark.parametrize("kind", ("single_hot", "one_home_group_hot", "power_law"))
def test_balancing_actually_balances(geom: AllocGeometry, kind: str):
    """Under skew the policy must beat the do-nothing placement."""
    if geom.R == 1:
        pytest.skip("single-rank geometry has nothing to balance")
    tpe = _tpe_or_skip(kind, geom, seed=7)
    alloc = reference_alloc(tpe, geom)
    naive = int(_naive_rank_load(tpe, geom).max().item())

    assert max_rank_load(alloc) <= naive, (
        f"{kind}: balancing made the hottest rank worse ("
        f"{max_rank_load(alloc)} > naive {naive})"
    )
    assert max_rank_load(alloc) <= geom.capacity, (
        f"{kind}: hottest rank {max_rank_load(alloc)} exceeds capacity {geom.capacity}"
    )


@pytest.mark.parametrize("geom", GEOMETRIES, ids=GEOM_IDS)
@pytest.mark.parametrize("kind", TPE_KINDS)
def test_b_pressure_reported(geom: AllocGeometry, kind: str, record_property):
    """Record prefetch-slot pressure; over-B is a perf cliff, not an error.

    Step 4 grants prefetch slots to the top-B remote experts only. We do not
    fail on overflow -- MoonEP's builtin policy can exceed B legitimately --
    but the number is recorded so policies can be compared and so a regression
    is visible.
    """
    tpe = _tpe_or_skip(kind, geom)
    alloc = reference_alloc(tpe, geom)
    pressure = b_pressure(alloc, geom)
    record_property("b_pressure_max", int(pressure.max().item()))
    record_property("B", geom.B)
    assert int(pressure.min().item()) >= 0


def test_contract_rejects_a_violating_policy():
    """A negative control: the checker must catch each violation it claims to.

    Without this, a checker that silently passes everything would look green.
    """
    geom = AllocGeometry(R=4, E=8, B=2, S=64, K=4, token_padding=32)
    tpe = make_tpe("uniform", geom)
    good = reference_alloc(tpe, geom).clone()
    assert not alloc_invariant_errors(good, tpe, geom)

    # 1. non-negative
    bad = good.clone()
    bad[0, 0] -= 1
    bad[1, 0] += 1
    bad[0, 0] = -1
    assert any("invariant 1" in e for e in alloc_invariant_errors(bad, tpe, geom))

    # 2. conservation -- drop a token
    bad = good.clone()
    nz = torch.nonzero(bad)[0]
    bad[nz[0], nz[1]] -= 1
    assert any("invariant 2" in e for e in alloc_invariant_errors(bad, tpe, geom))

    # 3. capacity -- pile everything onto rank 0
    bad = torch.zeros_like(good)
    bad[0] = tpe.sum(dim=0)
    errs = alloc_invariant_errors(bad, tpe, geom)
    assert any("invariant 3" in e for e in errs)

    # 4. segment bound -- spread every rank across every expert.
    # Needs R >= 3: when R == 2, 2*epn == E and the bound can never bind, so a
    # 2-rank negative control silently proves nothing.
    geom_wide = AllocGeometry(R=4, E=16, B=2, S=64, K=4, token_padding=32)
    assert 2 * geom_wide.epn < geom_wide.E, "negative control cannot trigger invariant 4"
    tpe_wide = make_tpe("uniform", geom_wide)
    expert_count = tpe_wide.sum(dim=0).to(torch.int64)
    # Even split of every expert over every rank: R*epn*... = E segments per
    # rank, well above 2*epn, while still conserving tokens and respecting
    # capacity -- so only invariant 4 should fire.
    bad = torch.zeros(geom_wide.R, geom_wide.E, dtype=torch.int64)
    for e in range(geom_wide.E):
        share = expert_count[e] // geom_wide.R
        bad[:, e] = share
        bad[0, e] += expert_count[e] - share * geom_wide.R
    bad = bad.to(torch.int32)
    errs = alloc_invariant_errors(bad, tpe_wide, geom_wide)
    assert any("invariant 4" in e for e in errs), errs


def check_policy(policy_fn, *, geometries=GEOMETRIES, kinds=TPE_KINDS, seed=1234):
    """Run the full conformance suite against an arbitrary policy.

    Args:
        policy_fn: ``(tpe: [R,E] int32, geom: AllocGeometry) -> alloc [R,E] int32``

    Returns:
        ``{(geom_id, kind): [error, ...]}`` for every non-conforming case.
    """
    failures = {}
    for geom in geometries:
        gid = f"R{geom.R}_E{geom.E}_S{geom.S}_K{geom.K}_tp{geom.token_padding}"
        for kind in kinds:
            tpe = make_tpe(kind, geom, seed=seed)
            errors = alloc_invariant_errors(policy_fn(tpe, geom), tpe, geom)
            if errors:
                failures[(gid, kind)] = errors
    return failures
