"""Planning-kernel microbenchmark, per allocation policy.

The allocation-policy hook only touches planning, so a full dispatch/combine
benchmark dilutes any regression it causes. This times ``launch_planning``
alone, which is the region under test.

Run with:
    torchrun --nproc_per_node=4 -m benchmarks.bench_planning
    torchrun --nproc_per_node=4 -m benchmarks.bench_planning --policies builtin,reference_greedy
"""

import argparse
import os

import torch
import torch.distributed as dist


CONFIGS = [
    # (name, S, H, K, epn, num_sms, token_padding, bias_ratio)
    ("balanced_epn16", 4096, 3584, 8, 16, 32, 128, 0.1),
    ("skewed_epn16", 4096, 3584, 8, 16, 32, 128, 1.0),
    ("degenerate_epn16", 4096, 3584, 8, 16, 32, 128, 5.0),
    ("balanced_epn32", 2048, 3584, 8, 32, 32, 128, 0.1),
    ("skewed_epn32", 2048, 3584, 8, 32, 32, 128, 1.0),
]

WARMUP = 20
ITERS = 100


def _make_topk(S, K, E, R, bias_ratio, rank, seed=1234):
    from tests.generate_topk_routing import generate_topk_routing

    return generate_topk_routing(S, K, E, R, bias_ratio, "cuda", seed, rank=rank)


def time_policy(policy, cfg, rank, R):
    from moonep import Buffer
    from moonep.planning import allocate_planning_outputs, launch_planning

    name, S, H, K, epn, num_sms, token_padding, bias = cfg
    E = R * epn

    kwargs = dict(num_sms=num_sms, token_padding=token_padding)
    # The pristine upstream Buffer has no alloc_policy argument; allowing it to
    # be omitted lets the identical script run against the baseline checkout,
    # which is what makes the A/B meaningful.
    import inspect
    if "alloc_policy" in inspect.signature(Buffer.__init__).parameters:
        kwargs["alloc_policy"] = policy
    elif policy != "builtin":
        raise SystemExit(f"this checkout has no alloc_policy support: {policy!r}")

    buffer = Buffer(S, H, K, E, R, **kwargs)
    try:
        ctx = buffer._require_ctx()
        topk, tpe = _make_topk(S, K, E, R, bias, rank)
        topk_flat = topk.reshape(-1).contiguous()
        plan, cu_seqlens = allocate_planning_outputs(ctx)

        for _ in range(WARMUP):
            launch_planning(ctx, topk_flat, tpe, cu_seqlens, plan)
        torch.cuda.synchronize()
        dist.barrier()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(ITERS):
            launch_planning(ctx, topk_flat, tpe, cu_seqlens, plan)
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / ITERS * 1000.0  # microseconds
    finally:
        buffer.destroy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", default="builtin")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    policies = [p for p in args.policies.split(",") if p]

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    R = dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    results = {}
    for cfg in CONFIGS:
        for policy in policies:
            us = time_policy(policy, cfg, rank, R)
            # the slowest rank sets the pace of a cooperative kernel
            t = torch.tensor([us], device="cuda")
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
            results[(cfg[0], policy)] = t.item()

    if rank == 0:
        tag = f" [{args.tag}]" if args.tag else ""
        print(f"\n=== planning kernel, EP={R}{tag} (us, max over ranks) ===")
        width = max(len(c[0]) for c in CONFIGS) + 2
        header = "config".ljust(width) + "".join(p.ljust(22) for p in policies)
        print(header)
        print("-" * len(header))
        for cfg in CONFIGS:
            row = cfg[0].ljust(width)
            base = results[(cfg[0], policies[0])]
            for policy in policies:
                v = results[(cfg[0], policy)]
                delta = (v / base - 1.0) * 100.0
                cell = f"{v:8.2f}" + (f" ({delta:+5.1f}%)" if policy != policies[0] else "")
                row += cell.ljust(22)
            print(row)
        print()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
