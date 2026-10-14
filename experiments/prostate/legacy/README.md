# Fixed-schedule and policy experiments (prostate)

These configs define the paper's fixed-schedule experiments and adaptive-policy variants. They
use an earlier protocol than the final BAP-MOS method, so compare them with BAP-MOS only with
that difference in mind. The final method is under `../bapmos/`.

| Folder | Experiments | In the paper |
|--------|-------------|--------------|
| `box/` | Box-only prompts | Main table (SAM row, method id `box`) |
| `point/` | Point-only prompts | Run alongside the fixed schedules |
| `box_point/` | Box:Point = 10/90, 50/50, 90/10 | Fixed-schedule table |
| `boxpoint_box_point/` | Box:Point:Both = 1:1:1, 1:5:1, 2:3:5, 5:1:1 | Fixed-schedule table |
| `policies/` | `ucb1_global`, `ucb1_per_organ`, `epsilon_greedy_per_organ`, `epsilon_decay_per_organ` | Adaptive policy variants |

## Protocol

- Reward (policies): negated validation MSD, clipped at 20 mm (`reward_clip_max_msd_mm: 20`)
- Negative-point ring: fixed `ring_width: 20` px (BAP-MOS scales the ring with organ size)
- Loss: CE + Dice. `bapmos.legacy.optimization.trainer` enforces this for the ratios and
  policies.
- Checkpointing: fold-balanced validation MSD on the PTV, as in BAP-MOS

BAP-MOS instead uses the composite reward, the organ-scaled ring and the boundary-weighted loss
([`docs/INNER_OUTER_LOOP.md`](../../../docs/INNER_OUTER_LOOP.md)).

## Running

- Box and point: `bapmos.multiorgan.train_sam_multiorgan_decoder_box` and
  `bapmos.multiorgan.train_sam_multiorgan_decoder_points`, with `--data_root`, `--run_root`,
  `--run_name` and `--seed` on the command line. Their `config.yaml` files record the intended
  values; the trainers do not read them.
- Ratios and policies: `bapmos.legacy.optimization.trainer --config <config.yaml>
  --experiment <name> --run_root runs/prostate/legacy --run_name <run_name>`. Ratio experiments
  are `box10_point90`, `box50_point50`, `box90_point10` and `ratio_*`; set `PI_TRAIN_SEED=43` or
  `44` for their replicates. Policy configs have one experiment per seed
  (`ucb1_global_seed42`, `ucb1_global_seed43_rep2`, …).

Full commands, test export and collation:
[`docs/RUNNING.md`](../../../docs/RUNNING.md#4-fixed-schedule-and-policy-experiments-prostate).
The trainers live in `src/bapmos/legacy/`.
