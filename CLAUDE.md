# second-order-conf

Research code for PLAN.md: a model forecasts `m ± s` for its own solve rate p_x, scored as a
Beta-Binomial prediction of k successes out of n rollouts. PLAN.md is the source of truth for
decisions, gates and the task list.

## Layout

- `src/soc/forecast/` — (m, s) ↔ (α, β), Beta-Binomial NLL, RPS, forecast/confidence parsers
- `src/soc/eval/` — PIT, interval coverage, ECE, Brier, AUROC, plots
- `src/soc/{data,verify,rollouts,head,rl}/` — per PLAN.md
- `configs/` — one YAML per experiment run
- `scripts/gcp/` — VM lifecycle, GCS sync, idle shutdown, job supervisor

## Commands

- `uv sync` then `uv run pytest` — tests (torch tests skip unless `uv sync --extra train`)
- `uv run ruff check . && uv run ruff format --check .`
- `uv run --extra data python -m soc.data.splits check` — re-verify the frozen splits against the Hub

## Rules

- Never launch, resize or delete GCP resources. Write the script, then stop and ask for approval.
- Every run records its config, git commit and seed in `run.json` and writes outputs to
  `gs://$SOC_BUCKET/<stage>/<run_id>/`. W&B is off for now (`wandb_project: null`).
- Train/val/test splits and the prompt template are frozen in week 1; their hashes are recorded
  in `configs/frozen.yaml`. Target rollouts and evaluation rollouts share template and
  temperature (0.7).
- Scoring code is unit-tested before any GPU time is spent. Do not change a scoring function
  without updating its tests.
- Long jobs checkpoint at least every 30 minutes and resume from the latest GCS checkpoint after
  spot pre-emption.
- Small, reviewable PRs, one task from the PLAN.md task list each.
- Secrets (W&B, judge API keys) live in Secret Manager or VM metadata, never in the repo.
