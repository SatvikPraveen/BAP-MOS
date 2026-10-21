"""Block-level prompt-selection history of one BAP-MOS run (run on GAIVI from the repo root).

train.log records every block's per-organ rewards (rounded to 4 decimals) but only the first
block's arms. The arms are recovered by replaying bapmos.method.bandit_policy with the logged
rewards. Where two arms' UCB scores are closer than the rounding can resolve, the replay
branches, and only a path that reproduces the bandit state saved in best_checkpoint.pth and
last_checkpoint.pth (t and N_a exactly; reward windows within the rounding) is accepted.
Output: JSON on stdout with per-block arms and rewards (no image ids).

    python scripts/camera_ready/results/bandit_dynamics.py runs/prostate/camera_ready/reference_seed42
"""
import ast
import copy
import json
import re
import sys
from pathlib import Path

import torch

from bapmos.method.bandit_policy import build_prompt_policy

TOL = 6e-5  # log rewards are rounded to 4 decimals
NEAR = 2e-4  # UCB scores closer than this may be reordered by that rounding

run = Path(sys.argv[1])
cfg = json.loads((run / "config.json").read_text())
b = cfg["bandit"]
organs, arms_all = list(b["organs"]), list(b["arms"])
log = (run / "logs" / "train.log").read_text().splitlines()

first = next(ast.literal_eval(l.split("New block arms: ", 1)[1]) for l in log if "New block arms: " in l)
blocks = []
for l in log:
    m = re.search(r"Block end epoch=(\d+) .* rewards=(\{.*\})\s*$", l)
    if m:
        blocks.append((int(m.group(1)), ast.literal_eval(m.group(2))))
ckpts = [torch.load(run / f"{k}.pth", map_location="cpu", weights_only=False)["bandit_state"]
         for k in ("best_checkpoint", "last_checkpoint")]
policy = build_prompt_policy(
    b.get("scope", "per_organ"), organs, arms=arms_all,
    window_size=int(b.get("window_size", 50)),
    warmup_blocks_per_arm=int(b.get("warmup_blocks_per_arm", 10)),
    min_pulls_per_arm=int(b.get("min_pulls_per_arm", 5)),
    seed=int(cfg.get("experiment_seed", 42)), memory=b.get("memory", "sliding"),
)


def matches(bd, s):
    """Replayed bandit equals a saved per-organ state."""
    return (bd.t == int(s["t"]) and dict(bd.N_a) == {a: int(n) for a, n in s["N_a"].items()}
            and all(len(bd._rewards[a]) == len(s["rewards"][a])
                    and all(abs(x - y) <= TOL for x, y in zip(bd._rewards[a], s["rewards"][a]))
                    for a in arms_all))


def consistent(bd, targets):
    """After a commit: every saved state still reachable (N_a not exceeded, and a commit that
    falls inside a saved reward window has that window's reward)."""
    for t, s in targets.items():
        if bd.t > t:
            continue
        for a in arms_all:
            n_final, w = int(s["N_a"][a]), s["rewards"][a]
            if bd.N_a[a] > n_final:
                return False
            pos = bd.N_a[a] - (n_final - len(w)) - 1  # index of the latest commit in the window
            if pos >= 0 and bd._last_commit[0] == a and abs(bd._last_commit[1] - w[pos]) > TOL:
                return False
    return True


def commit(bd, arm, reward):
    bd.commit_block_reward(arm, reward)
    bd._last_commit = (arm, reward)


def replay_organ(organ):
    """(arm per block, paths explored) for one organ's bandit."""
    seq = [r.get(organ) for _, r in blocks]
    targets = {int(s["organs"][organ]["t"]): s["organs"][organ] for s in ckpts}
    stack, explored = [(0, copy.deepcopy(policy.bandits[organ]), [])], 0
    while stack:
        i, bd, path = stack.pop()
        explored += 1
        ok = True
        while ok and i < len(seq):
            if seq[i] is None:  # organ had no probe reward in this block
                path, i = path + [None], i + 1
                continue
            options = [bd.select_arm()]
            if bd.t >= bd.warmup_blocks_per_arm * len(bd.arms) and all(
                    bd.N_a[a] >= bd.min_pulls_per_arm for a in bd.arms):
                scores = {a: bd._ucb_score(a) for a in bd.arms}
                options += [a for a in bd.arms if a != options[0] and scores[options[0]] - scores[a] <= NEAR]
            for alt in options[1:]:  # alternatives are explored after the replayed choice fails
                b2 = copy.deepcopy(bd)
                commit(b2, alt, seq[i])
                if consistent(b2, targets):
                    stack.append((i + 1, b2, path + [alt]))
            commit(bd, options[0], seq[i])
            path, i = path + [options[0]], i + 1
            ok = consistent(bd, targets) and (bd.t not in targets or matches(bd, targets[bd.t]))
        if ok and all(t <= bd.t for t in targets):
            return path, explored
    sys.exit(f"{organ}: no replay path reproduces the checkpoints ({explored} paths)")


paths, explored = {}, {}
for o in organs:
    paths[o], explored[o] = replay_organ(o)
rows = [{"block": i + 1, "epoch": epoch, "arms": {o: paths[o][i] for o in organs if paths[o][i]}, "rewards": rewards}
        for i, (epoch, rewards) in enumerate(blocks)]
if rows[0]["arms"] != first:
    sys.exit(f"replay: first block {rows[0]['arms']} != logged {first}")
print(json.dumps({
    "run": run.name, "n_blocks": len(rows),
    "warmup_blocks": int(b.get("warmup_blocks_per_arm", 10)) * len(arms_all),
    "replay_paths_explored": explored,
    "check": "replay reproduces best_checkpoint and last_checkpoint bandit_state (t, N_a exact; windows within 6e-5) and the logged first block",
    "blocks": rows,
}))
