"""The allocation-policy contract: invariants, adversarial cases, and a driver.

``alloc`` is the seam between MoonEP's communication backend and a pluggable
allocation policy. It is an int32 ``[R, E]`` matrix (the kernel's layout, flat
as ``alloc[d * E + e]``) where ``alloc[d, e]`` is the number of *global* tokens
routed to expert ``e`` that are computed on destination rank ``d``.

Everything downstream of the policy -- ``expert_off``, ``cu_seqlens``,
``experts_to_copy``, ``zero_fill_ranges``, ``remote_stats`` and ``dst`` -- is a
pure function of ``alloc`` (see ``planning_reference.py`` part 2). A policy is
therefore free to choose any ``alloc`` satisfying the invariants below, and the
backend will build a correct plan from it.

This module is imported by MoonEP's own tests and is intended to be imported by
out-of-tree policy implementations (e.g. MLB) as the shared conformance suite.
"""

from dataclasses import dataclass

import torch


# ============================================================
# Invariants
# ============================================================
#
# 1. NON-NEGATIVE       alloc >= 0
# 2. CONSERVATION       alloc.sum(dim=0) == expert_count
#                       every routed token is computed exactly once
# 3. CAPACITY           alloc.sum(dim=1) <= NvS_capacity  (= S * K)
#                       no destination rank exceeds its real token capacity
# 4. SEGMENT BOUND      (alloc[d] > 0).sum() <= 2 * epn   for every d
#                       *** the implicit one ***
#
# Invariant 4 is not stated anywhere upstream, but it is load-bearing: the
# dispatch buffer is sized in api.py as
#
#     token_padding_extra = (token_padding - 1) * 2 * epn
#     NvS = NvS_capacity + token_padding_extra
#
# i.e. the padding headroom budgets for at most ``2 * epn`` *non-empty* VM
# groups per destination rank, each of which can waste up to
# ``token_padding - 1`` slots rounding up to a segment boundary. A policy that
# spreads one destination rank's inbound tokens over more than ``2 * epn``
# distinct experts overflows NvS and trips the
# "padded layout exceeds NvS" assertion -- or, in the fused kernel, corrupts
# memory past the buffer.
#
# MoonEP's builtin greedy policy satisfies invariant 4 by construction: its
# balance loop sets ``balance[u] = 0`` when it selects receiver ``u``, so ``u``
# is never selected again. Every destination rank therefore receives from *at
# most one* home group, giving it at most ``epn`` own experts plus at most
# ``epn`` received experts. Any replacement policy must reproduce this
# structural property, not merely the load balance.
#
# 5. PADDED FIT         the step-4 padded layout must fit in NvS
#                       implied by 3 + 4, checked directly as a backstop
#
# The following is *not* an invariant but a quality metric, reported by the
# checker so policies can be compared:
#
#   B-PRESSURE          per dest rank, the number of non-empty *remote* expert
#                       groups. Step 4 grants prefetch slots to only the top-B
#                       of these (planning_reference.py:158). Beyond B, remote
#                       experts still work but are not weight-prefetched, which
#                       is a silent throughput cliff rather than an error.


@dataclass(frozen=True)
class AllocGeometry:
    R: int
    E: int
    B: int
    S: int
    K: int
    token_padding: int = 128

    @property
    def epn(self) -> int:
        return self.E // self.R

    @property
    def capacity(self) -> int:
        return self.S * self.K

    @property
    def NvS(self) -> int:
        return self.capacity + (self.token_padding - 1) * 2 * self.epn

    def ctx(self, rank: int = 0) -> dict:
        return {
            "rank": rank,
            "R": self.R,
            "E": self.E,
            "B": self.B,
            "S": self.S,
            "K": self.K,
            "NvS_capacity": self.capacity,
            "NvS": self.NvS,
            "token_padding": self.token_padding,
            "group": None,
        }


def alloc_invariant_errors(alloc, tpe, geom: AllocGeometry) -> list[str]:
    """Return a list of contract violations; empty means conforming.

    Args:
        alloc: int32 [R, E] -- the policy's output, kernel layout.
        tpe: int32 [R, E] -- tokens per expert per *source* rank.
        geom: the buffer geometry the policy was invoked under.
    """
    errors: list[str] = []
    alloc = alloc.cpu().to(torch.int64)
    tpe = tpe.cpu().to(torch.int64)
    R, E, epn = geom.R, geom.E, geom.epn

    if tuple(alloc.shape) != (R, E):
        return [f"shape {tuple(alloc.shape)} != expected {(R, E)}"]

    # 1. non-negative
    if bool((alloc < 0).any()):
        bad = torch.nonzero(alloc < 0)[:4].tolist()
        errors.append(f"invariant 1 (non-negative): negative entries at {bad}")

    # 2. conservation
    expert_count = tpe.sum(dim=0)
    got = alloc.sum(dim=0)
    if not torch.equal(got, expert_count):
        bad = torch.nonzero(got != expert_count).flatten()[:4].tolist()
        errors.append(
            f"invariant 2 (conservation): experts {bad} have "
            f"alloc.sum={got[bad].tolist()} != expert_count={expert_count[bad].tolist()}"
        )

    # 3. capacity
    per_rank = alloc.sum(dim=1)
    if bool((per_rank > geom.capacity).any()):
        bad = torch.nonzero(per_rank > geom.capacity).flatten()[:4].tolist()
        errors.append(
            f"invariant 3 (capacity): ranks {bad} hold {per_rank[bad].tolist()} "
            f"> capacity {geom.capacity}"
        )

    # 4. segment bound -- the implicit NvS-sizing assumption
    segments = (alloc > 0).sum(dim=1)
    if bool((segments > 2 * epn).any()):
        bad = torch.nonzero(segments > 2 * epn).flatten()[:4].tolist()
        errors.append(
            f"invariant 4 (segment bound): ranks {bad} have "
            f"{segments[bad].tolist()} non-empty groups > 2*epn={2 * epn}; "
            f"this overflows the NvS padding budget set in api.py"
        )

    # 5. padded fit -- direct backstop for 3 + 4
    tp = geom.token_padding
    padded = ((alloc + tp - 1) // tp * tp * (alloc > 0)).sum(dim=1)
    if bool((padded > geom.NvS).any()):
        bad = torch.nonzero(padded > geom.NvS).flatten()[:4].tolist()
        errors.append(
            f"invariant 5 (padded fit): ranks {bad} need {padded[bad].tolist()} "
            f"padded slots > NvS={geom.NvS}"
        )

    return errors


def b_pressure(alloc, geom: AllocGeometry) -> torch.Tensor:
    """Per dest rank, the count of non-empty *remote* expert groups.

    Values above ``geom.B`` mean step 4 cannot prefetch every remote expert's
    weights (planning_reference.py:158 keeps only the top-B by token count).
    """
    alloc = alloc.cpu()
    R, E, epn = geom.R, geom.E, geom.epn
    out = torch.zeros(R, dtype=torch.int32)
    for d in range(R):
        local = range(d * epn, (d + 1) * epn)
        out[d] = sum(
            1 for e in range(E) if alloc[d, e].item() > 0 and e not in local
        )
    return out


def max_rank_load(alloc) -> int:
    """The objective the policy is minimizing: the hottest rank's token count."""
    return int(alloc.cpu().to(torch.int64).sum(dim=1).max().item())


# ============================================================
# Adversarial cases
# ============================================================


def make_tpe(kind: str, geom: AllocGeometry, seed: int = 0) -> torch.Tensor:
    """Build a ``[R, E]`` tokens-per-expert matrix with a chosen pathology.

    Every case must satisfy ``tpe.sum(dim=1) == S * K`` per source rank -- that
    is a hard property of top-k routing, not a policy choice, and the balance
    loop's ``sum(balance) == 0`` depends on it.
    """
    R, E, N = geom.R, geom.E, geom.capacity
    g = torch.Generator().manual_seed(seed)

    if kind == "uniform":
        base = torch.full((R, E), N // E, dtype=torch.int64)
        rem = N - (N // E) * E
        if rem:
            base[:, :rem] += 1
        return base.to(torch.int32)

    if kind == "single_hot":
        # every rank sends everything to one expert -- maximal imbalance
        tpe = torch.zeros(R, E, dtype=torch.int64)
        tpe[:, 0] = N
        return tpe.to(torch.int32)

    if kind == "one_home_group_hot":
        # all tokens land on rank 0's home experts: rank 0 must shed to all
        # others. This is the case that stresses invariant 4 hardest.
        tpe = torch.zeros(R, E, dtype=torch.int64)
        epn = geom.epn
        per = N // epn
        tpe[:, :epn] = per
        tpe[:, 0] += N - per * epn
        return tpe.to(torch.int32)

    if kind == "one_rank_hot":
        # only source rank 0 is skewed; the rest are uniform
        tpe = make_tpe("uniform", geom, seed).to(torch.int64)
        tpe[0] = 0
        tpe[0, 0] = N
        return tpe.to(torch.int32)

    if kind == "random":
        tpe = torch.zeros(R, E, dtype=torch.int64)
        for r in range(R):
            idx = torch.randint(0, E, (N,), generator=g)
            tpe[r] = torch.bincount(idx, minlength=E)
        return tpe.to(torch.int32)

    if kind == "power_law":
        w = 1.0 / torch.arange(1, E + 1, dtype=torch.float64)
        tpe = torch.zeros(R, E, dtype=torch.int64)
        for r in range(R):
            idx = torch.multinomial(w, N, replacement=True, generator=g)
            tpe[r] = torch.bincount(idx, minlength=E)
        return tpe.to(torch.int32)

    if kind == "intra_group_skew":
        # Wildly unequal *experts*, exactly equal *ranks*: all skew is confined
        # within each home group, so group_tokens stays at capacity on every
        # rank and no migration is ever justified. This separates the two
        # objectives a policy could confuse -- balancing experts (wrong) versus
        # balancing ranks (right).
        if N % E != 0:
            raise ValueError("intra_group_skew requires E to divide S*K")
        epn = geom.epn
        tpe = torch.zeros(R, E, dtype=torch.int64)
        for r in range(R):
            for h in range(R):
                lo, hi = h * epn, (h + 1) * epn
                total = (N // E) * epn
                if epn == 1:
                    tpe[r, lo] = total
                    continue
                # dump the whole home group's quota onto one rotating expert
                pick = lo + (r + h) % epn
                tpe[r, lo:hi] = 0
                tpe[r, pick] = total
        return tpe.to(torch.int32)

    if kind == "empty_experts":
        # half the experts are never routed to
        tpe = torch.zeros(R, E, dtype=torch.int64)
        live = max(1, E // 2)
        per = N // live
        tpe[:, :live] = per
        tpe[:, 0] += N - per * live
        return tpe.to(torch.int32)

    raise ValueError(f"unknown tpe kind {kind!r}")


TPE_KINDS = (
    "uniform",
    "single_hot",
    "one_home_group_hot",
    "one_rank_hot",
    "random",
    "power_law",
    "empty_experts",
    "intra_group_skew",
)

# Kinds that require E | S*K; skipped on geometries where it does not divide.
TPE_KINDS_NEEDING_DIVISIBLE = frozenset({"intra_group_skew"})


GEOMETRIES = (
    AllocGeometry(R=8, E=128, B=8, S=256, K=8),
    AllocGeometry(R=8, E=8, B=2, S=64, K=4, token_padding=32),
    AllocGeometry(R=2, E=2, B=1, S=3, K=5, token_padding=32),
    AllocGeometry(R=4, E=4, B=1, S=33, K=2, token_padding=1),
    AllocGeometry(R=8, E=256, B=16, S=1024, K=8),
    AllocGeometry(R=1, E=16, B=0, S=128, K=4),
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
