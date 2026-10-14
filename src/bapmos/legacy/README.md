# Legacy library modules

Historical trainers and helpers for prompt/policy ablations under
`experiments/*/legacy/`. **Not required** to reproduce main BAP-MOS results
(`experiments/*/bapmos/` + `experiments/*/baselines/`).

The legacy *trainers* are optional: they are needed only for the fixed-schedule and policy
experiments. `bapmos.legacy.optimization.metrics` is not optional: the main evaluation path
(validation and checkpoint metrics in `bapmos.metrics`) imports its `MetricsEvaluator`.

See `experiments/prostate/legacy/README.md` for the legacy protocol note.
