# Second-Order Confidence Research Plan

Oct 20, 2026 · @Henry Maguire

## Overview

We will test whether a language model can state a calibrated, useful *spread* around its own solve rate, not just a point confidence, aiming for a workshop paper with all code and data released.

**The idea.** Before reasoning, the model writes `<forecast> m ± s </forecast>`: a belief about p\_x, its own probability of solving question x at its sampling temperature. After answering, it writes the usual RLCR-style `<confidence> q </confidence>` about that specific answer. A single right/wrong outcome cannot train a spread, but a group of n rollouts gives k successes, and k does. We convert (m, s) to a Beta distribution and score the implied Beta-Binomial prediction of k with a proper score.

**Hypotheses.**

1. The spread is calibrated: observed k falls where the forecast says it should (flat PIT, correct interval coverage).
2. The spread is informative beyond m: a per-question spread beats one global spread, and is not just a function of predicted difficulty.
3. The spread grows under distribution shift.
4. Adding it costs nothing in accuracy or answer-level calibration relative to CALIBER.

**Approach.** Three stages, each ending in a go/no-go gate: Stage 0 trains a small head on a frozen model (no RL); Stage 1 teaches a LoRA forecaster to write the spread as text; Stage 2 trains everything jointly with per-token reward routing and compares against RLVR, RLCR and a CALIBER replication.

**Closest prior work.** CALIBER trains a pre-thinking point confidence against the group solve rate; nothing we found trains a verbalised second-order belief scored against k-of-n. A proper literature sweep is the first task.

## Decisions

All 21 setup questions use the suggested defaults; change any row here and the rest of the plan follows.

| # | Topic | Decision |
| --- | --- | --- |
| 1 | Goal | Workshop paper; release code, data and weights |
| 2 | Scope | All three stages, go/no-go gate after each |
| 3 | Budget, timeline | ≤ $500 GPU for Stages 0–1; Stage 2 ceiling $2,000 (assumed, needs your OK); 4–6 weeks to the Stage 1 gate |
| 4 | Success | Concrete gate thresholds in Evaluation |
| 5 | Model | Qwen3.5-2B instruct for Stages 0–1; next size up (\~4B if one exists) for Stage 2 |
| 6 | Reasoning | Instruct model, short reasoning (thinking mode off), max 2,048 new tokens; raised from 1,024 after the smoke test cut off 51% of BigMath rollouts |
| 7 | Training data | BigMathDigits (RLCR/CALIBER split) |
| 8 | OOD evaluation | GPQA Diamond, TriviaQA, SimpleQA |
| 9 | Verification | Programmatic for training; cheap LLM judge for TriviaQA/SimpleQA evaluation only |
| 10 | Rollouts | n = 32 per question for both training targets and evaluation, temperature 0.7 |
| 11 | Stage 0 head | Linear and 2-layer MLP on the last-layer state at the final prompt token |
| 12 | Baselines | Point-estimate head, global-spread head, difficulty-only spread; K-sample spread in Stage 2 |
| 13 | Loss | Beta-Binomial NLL in Stage 0; bounded RPS in the RL stages |
| 14 | Verbal format | `m ± s`, two decimals |
| 15 | CALIBER baseline | Re-implemented at our scale |
| 16 | GPUs | L4 for Stage 0, A100 (H100 if quota) for RL; request quota on day 1 |
| 17 | Pricing | Spot VMs, checkpoints to GCS, resumable jobs |
| 18 | Environment | Single Compute Engine VM, Deep Learning VM image |
| 19 | Frameworks | vLLM + PyTorch/HF (Stage 0), TRL or Unsloth (Stage 1), verl (Stage 2), W&B, uv |
| 20 | Claude Code autonomy | Writes code and launch scripts; you approve every VM launch; idle auto-shutdown |
| 21 | Repo, release | New GitHub repo, Apache 2.0, release everything |

## Repo and conventions

One Python package, one config per experiment, every artifact in a GCS bucket, and tests for every scoring function before any GPU time is spent.

```
second-order-conf/
  CLAUDE.md            # project rules for Claude Code (this section, condensed)
  PLAN.md              # this document, exported
  pyproject.toml       # uv-managed
  configs/             # one YAML per experiment run
  src/soc/
    data/              # dataset loaders, splits, prompt templates
    verify/            # math-verify, exact match, LLM judge
    rollouts/          # vLLM generation -> per-question k, n (parquet)
    head/              # feature extraction, Stage 0 heads, Laplace option
    forecast/          # parse "m ± s", Beta conversion, Beta-Binomial NLL, RPS
    rl/                # Stage 1 (TRL/Unsloth), Stage 2 (verl patch)
    eval/              # PIT, coverage, ECE, Brier, AUROC, plots
  scripts/gcp/         # create/start/stop VM, sync to GCS, idle shutdown
  tests/
```

**Rules for Claude Code**

- Never launch, resize or delete GCP resources; write the script and stop for approval.
- Every run logs its config, git commit and seed to W&B, and writes outputs to `gs://<bucket>/<stage>/<run_id>/`.
- Freeze the train/val/test splits and the prompt template in the first week; record their hashes. Rollouts used as targets and rollouts used for evaluation must share template and temperature.
- Unit-test the scoring code first: Beta-Binomial NLL against `scipy.stats.betabinom`, RPS bounded in \[−1, 0\], the (m, s) → (α, β) conversion and its clamps, and the forecast parser on malformed strings.
- Long jobs checkpoint at least every 30 minutes and resume from the latest checkpoint after a spot pre-emption.
- Prefer small, reviewable PRs, one task from the task list each.

## GCP setup

Request GPU quota on day one: new projects often start with none for A100s and H100s, and approval can take days.

1. **Quota.** In IAM & Admin → Quotas, request the global GPU quota plus regional quotas for L4, A100 and H100 in one or two regions with stock. Spot (preemptible) GPU quotas are separate from on-demand; request both.
2. **Budget alerts.** In Billing, set a budget for the project with alerts at 50%, 90% and 100% of the stage ceiling.
3. **Storage.** One regional GCS bucket in the same region as the VMs for datasets, rollouts, checkpoints and results.
4. **VMs.** Compute Engine with a Deep Learning VM image (PyTorch + CUDA). Stage 0: one G2 VM (1× L4, 24 GB), enough for vLLM on a 2B model. Stages 1–2: one A2 VM (1× A100), or A3 (H100) if quota allows. Spot provisioning by default.
5. **Pre-emption.** Jobs run under a small supervisor script that restarts them from the latest GCS checkpoint when the VM comes back.
6. **Idle shutdown.** A cron job on each VM stops it after 30 minutes with GPU utilisation under 5% and no training process running.
7. **Secrets.** W&B and LLM-judge API keys live in Secret Manager or VM metadata, never in the repo.

Prices vary by region and change often; check the GCP pricing calculator before each stage rather than trusting the estimates below.

## Stage 0: does the spread mean anything?

With the answering model frozen, each question's solve rate is fixed, so the core claim becomes supervised learning on a fixed dataset: no RL, one L4, tens of dollars.

**Data.** BigMathDigits: about 5,000 training questions, 1,000 validation, and the 1,000-question evaluation split used by RLCR and CALIBER. Get it from the RLCR release if available; otherwise rebuild it as RLCR describes (Big-Math, numeric answers, Llama-8B solve rate 0–70%). OOD: GPQA Diamond (198), a 1,000-question TriviaQA subset, a 1,000-question SimpleQA subset.

**Rollouts.** Qwen3.5-2B instruct, temperature 0.7, max 2,048 new tokens, with vLLM. n = 32 for every question (training, validation, test and OOD), so training targets and evaluation see the same n. Store every rollout (answer, correct, length) and per-question k, n as parquet in GCS.

**Features.** Hidden state at the final prompt token, before any generated token: last layer by default, plus one middle layer as an ablation. Cache them once to disk.

**Heads**, all trained with Beta-Binomial NLL against k out of n except the first:

| Head | What it tests |
| --- | --- |
| Point head (m only, squared error to k/n) | Baseline point forecast, as in CALIBER |
| Point head + one global concentration fitted on validation | Is any per-question spread needed at all? |
| Difficulty-only spread (concentration as a function of m) | Is the spread just predicted difficulty? |
| Linear (α, β) head | Main method |
| 2-layer MLP (α, β) head | Capacity check |
| Linear head + Laplace on its weights (optional) | Adds epistemic variance; compare with the learned spread |

The key comparison is row 4 against rows 2 and 3: the learned spread must beat a single global spread and a difficulty-only spread on held-out NLL.

**Usefulness test (empirical Bayes).** Treat each forecast as a prior, observe r ∈ {1, 2, 4, 8} fresh rollouts, and measure the error of the posterior mean against the 32-rollout solve rate (the r rollouts are drawn on top of those 32). If per-question spreads are right, they weight prior against evidence better than a global spread, so posterior error drops. A second test allocates a fixed budget of extra rollouts by forecast variance instead of uniformly.

## Stage 1: say the spread in words

A LoRA forecaster on the same base model learns to write `<forecast> m ± s </forecast>` for the frozen model's solve rate, reusing Stage 0's stored k values, so each sample is about 10 tokens.

**Roles.** One base model in memory. Forecaster = base + LoRA (rank 16, attention and MLP projections). Answerer = base with the adapter off; it does not run during Stage 1 because k is precomputed.

**Prompt.** Exactly the Stage 0 rollout prompt, followed by `<forecast>`. Generation stops at `</forecast>`.

**1a. SFT warm start.** Targets are the best Stage 0 head's (m, s) written as text, e.g. `0.62 ± 0.14`. One or two epochs on the training split.

**1b. RL refinement.** GRPO with TRL or Unsloth: 8 forecasts per question, temperature 1.0, max 16 new tokens. Reward is the negative RPS of the forecast's Beta-Binomial against the stored k out of 32; unparseable output gets −1. Advantages are mean-centred without dividing by the standard deviation, as in RLCR and CALIBER. Log the within-group spread of forecasts every step; if it collapses toward zero, raise the temperature or extend SFT.

**1c. Cross-model forecaster.** Repeat 1a–1b with a different, smaller model (Qwen3.5-0.8B) and one model from another family, all predicting Qwen3.5-2B's solve rates. If they match the self-forecaster, the spread comes from reading question difficulty rather than self-knowledge. Either result is reportable.

The RPS for a forecast with Beta-Binomial CDF F over counts 0..n and observed count k:

```latex
\mathrm{RPS} = \frac{1}{n+1} \sum_{j=0}^{n} \big( F(j) - \mathbf{1}[j \ge k] \big)^2, \qquad \alpha + \beta = \frac{m(1-m)}{s^2} - 1
```

## Stage 2: train it jointly

One policy forecasts its own solve rate, reasons, answers and states answer confidence, with each reward routed only to the tokens it should shape; the main comparison is against a CALIBER replication at the same scale.

**Output format**

```
<forecast> m ± s </forecast>
<think> ... </think>
<answer> ... </answer>
<analysis> ... </analysis>
<confidence> q </confidence>
```

**Rewards and routing.** Each reward gets its own group-centred advantage, applied only to its token span. This blocks a self-fulfilling forecast (answering worse to match a low forecast) and stops the low-variance forecast reward being drowned out, the collapse CALIBER reported.

| Reward | Formula | Applied to |
| --- | --- | --- |
| Correctness | cᵢ | think, answer |
| Answer calibration (Brier) | −(qᵢ − cᵢ)² | analysis, confidence |
| Forecast | −RPS of (mᵢ, sᵢ) against k₋ᵢ of n − 1 | forecast |
| Format | 1 if every tag is present and parseable | all tokens |

Here k₋ᵢ counts successes among the other rollouts in the group, so a rollout's own outcome never enters its forecast target. Implement routing in verl by replacing its advantage computation with per-span advantages built from tag masks.

**Training.** Base: the next Qwen3.5 size up from 2B (verify sizes; fall back to 2B if budget is tight). LoRA. Start from the Stage 1 forecaster adapter, then 50–100 forecast-only warmup steps. 16 rollouts per prompt (32 if budget allows), about 128 prompts per step, about 200 steps, max 2,048 new tokens, no KL penalty, no std normalisation.

**Runs** (same data, steps and budget):

| Run | Purpose |
| --- | --- |
| RLVR | Accuracy reference, no calibration |
| RLCR | Post-answer confidence only |
| CALIBER replication | Pre point confidence + post confidence, rewards summed, warmup |
| Ours, summed rewards | Isolates the effect of routing |
| Ours, routed rewards | Main method; 3 seeds if budget allows |

At evaluation, also compute the K-sample spread from the RLCR model (8 resampled analyses per answer) as a training-free baseline.

**Sandbagging check.** Compare per-question solve rates of the summed and routed runs against RLVR, split by forecast m. A summed run whose low-forecast questions lose accuracy relative to RLVR is fulfilling its forecasts.

## Evaluation and gates

Every forecast is judged against 32 fresh rollouts per held-out question (the same n as training) at temperature 0.7, with 95% bootstrap intervals over questions on every number.

**Forecast metrics**

- Beta-Binomial NLL and RPS on held-out k.
- Randomised PIT histogram (10 bins) of k under the forecast's Beta-Binomial; flat means calibrated, U-shaped means spreads too narrow, humped means too wide.
- Coverage of central 50%, 80% and 95% predictive intervals for k (intervals on counts, so binomial noise is already included).
- Sharpness: mean s. Shift: mean s on each OOD set versus in-domain.
- Usefulness: the empirical-Bayes posterior error at r = 1, 2, 4, 8 extra rollouts.

**Answer metrics** (Stage 2): accuracy, ECE (10 bins), Brier, AUROC of the post-answer confidence.

**Gates.** These thresholds are proposals; adjust before Stage 0 results arrive, not after.

| Gate | Pass if | If it fails |
| --- | --- | --- |
| Stage 0 → 1 | Linear (α, β) head beats both the global-spread and difficulty-only heads on held-out NLL (95% CI of the difference excludes 0); 80% interval coverage in 75–85%; no PIT bin more than 30% from uniform; empirical-Bayes error lower than with a global spread at r = 4 | Stop or pivot: write up the negative result, or test the Laplace variance as the spread instead |
| Stage 1 → 2 | Verbal forecaster's held-out RPS within 10% of the Stage 0 head; coverage 75–85%; ≥ 99% parseable; within-group forecast spread above zero throughout RL | Stop at Stage 1 as the paper's main result |
| Stage 2 result | Forecast meets the Stage 0 calibration bar; post-answer ECE within 0.01 of the CALIBER replication; accuracy within 2 points of RLVR; mean s higher on OOD than in-domain | Report what held; routed vs summed and the sandbagging check stand on their own |

## Budget, timeline and risks

Stages 0 and 1 should fit in about four weeks and under $500; Stage 2 is where cost and schedule can stretch.

&#91;embedded content: stage roadmap · 4 phases, 2 go/no-go gates\]

Equal-width bands, not to scale: a failed gate ends spending on the next stage, so the expensive Stage 2 runs only on evidence from the cheap ones.

| Stage | Weeks | GPU budget (order of magnitude) | Main cost |
| --- | --- | --- | --- |
| Setup + literature sweep | 0–1 | \~$0 | Quota approval time |
| Stage 0 | 1–2 | tens of dollars | One-off rollout generation |
| Stage 1 | 3–4 | low hundreds | GRPO on short forecasts, cross-model runs |
| Stage 2 | 5–8+ | up to $2,000 (assumed ceiling) | Five runs of full-length rollouts |

**Risks**

| Risk | Mitigation |
| --- | --- |
| GPU quota denied or delayed | Request on day 1; Stage 0 runs on an L4, which is easier to get |
| Spread is just predicted difficulty | Difficulty-only and cross-model baselines detect it; report it either way |
| Forecasts collapse to a few values in RL | SFT warm start, forecast-only warmup, per-span advantages, monitor within-group spread |
| Spread calibrated but uselessly wide | Sharpness and empirical-Bayes metrics catch it at the Stage 0 gate |
| Self-fulfilling forecasts in joint training | Per-token routing; summed-reward run plus sandbagging check |
| Someone publishes this first | Literature sweep in week 0; re-check citations of CALIBER before writing |
| verl modification takes longer than planned | Isolate it as one task with its own tests; Stage 1 results stand without it |
| LLM-judge cost on SimpleQA/TriviaQA | 1,000-question subsets, a cheap judge model, cache every verdict |

## Notes from the literature sweep

Considerations from the Week 0 sweep ([docs/related_work.md](docs/related_work.md)), recorded for later decisions; they are not tasks and nothing above has been changed to adopt them. The novelty claim holds in its narrow form: no paper trains a verbalised distribution over the model's own solve rate scored against k of n. The closest threat is CoCA (arXiv 2603.05881), which already has a pre-answer confidence against the group solve rate with segmented advantages.

1. **CoCA as a baseline.** CoCA is the most direct pre-answer point-confidence competitor, and predates CALIBER; consider running it alongside the CALIBER replication. Its exact confidence reward comes from secondary summaries, so read the paper before re-implementing it.
2. **Routing is prior art.** Per-span rewards and advantages appear in CoCA, DCPO, Blockwise Advantage Estimation and MRTPO. Present routing as reused machinery; the contribution is the summed-vs-routed comparison and the sandbagging check.
3. **Non-hackable reward check.** arXiv 2607.04332 formalises "confidence reward hacking" and gives a criterion for non-hackable rewards. Worth checking whether deliberately lowering p_x can ever raise the expected summed reward (RPS forecast + correctness), and reporting it next to the empirical sandbagging check.
4. **Forecast collapse.** CalibSFT (2609.32470) and CREDO (2609.34857) report verbal confidences collapsing onto a few values during RL. The SFT warm start should cover the (m, s) range broadly; consider also logging the number of distinct (m, s) values per step.
5. **No std normalisation on the forecast span.** arXiv 2508.11800 shows GRPO std normalisation biases probabilistic outputs; this supports the plan's mean-only advantages, and could be an ablation.
6. **Extra Stage 0 baselines to consider.** The pre-generation linear probe from "LLMs Encode Their Failures" (2602.09924) with its length and TF-IDF surface baselines; a peer-model probe ("Masked by Consensus", 2604.12373) to separate self-knowledge from shared difficulty; optionally a Future Confidence Distillation head (2607.07626).
7. **Capability calibration.** arXiv 2602.13540 names the metric for m against p_x; reporting it would make the point-forecast part comparable with prior work.
8. **Identifiability.** With n rollouts per question, only the first n moments of p_x are identified (2609.09245). Claims about s should stay within what the Beta-Binomial on n = 32 tests: PIT on k, coverage, overdispersion.
9. **Framing against Bengs, Hüllermeier and Waegeman.** Their result says no loss makes a learner report epistemic uncertainty faithfully from one label per input. It does not bite here, because each question has n > 1 rollouts and Binomial overdispersion identifies the Beta spread. Describe s as second-order uncertainty about one's own solve rate, not as "epistemic" uncertainty.

## Tasks for Claude Code

Hand these over one at a time, in order; each ends in a reviewable PR, and tasks marked (you) need a human.

**Week 0: setup**

- [ ] (you) Request GPU quotas, create the GCS bucket, set billing alerts, create W&B project
- [x] Scaffold the repo: uv project, package layout, CLAUDE.md from "Repo and conventions", CI running tests
- [x] Scoring library with tests: (m, s) ↔ (α, β), Beta-Binomial NLL, RPS, forecast parser, randomised PIT, interval coverage
- [x] GCP scripts: create/start/stop spot VM, GCS sync, idle-shutdown cron, resumable job supervisor
- [x] Literature sweep: citations of CALIBER, RLCR, COREA, CoCA and DCPO, plus searches for second-order calibration and pass-rate prediction; write a related-work note

**Stage 0**

- [x] Dataset loaders and frozen splits for BigMathDigits, GPQA Diamond, TriviaQA and SimpleQA subsets; prompt template; verifiers (math-verify, exact match, cached LLM judge)
- [x] vLLM rollout job: n rollouts per question, per-question k, parquet to GCS, resumable
- [ ] (you) Launch rollout jobs
- [ ] Feature extraction: final-prompt-token hidden states, last and middle layer, cached
- [ ] Train the six heads; log held-out NLL, RPS, PIT, coverage, sharpness, OOD shift
- [ ] Empirical-Bayes and adaptive-allocation usefulness tests
- [ ] Stage 0 report against the gate; (you) go/no-go

**Stage 1**

- [ ] SFT data from the best head; LoRA SFT of the forecaster
- [ ] GRPO forecaster training with RPS reward and within-group spread logging
- [ ] Cross-model forecasters (Qwen3.5-0.8B and one other family)
- [ ] Stage 1 report against the gate; (you) go/no-go

**Stage 2**

- [ ] verl setup with the five-tag format and reward functions
- [ ] Per-span advantage routing in verl, with a unit test on a toy batch
- [ ] Small-scale smoke run (20 steps) of every configuration
- [ ] (you) Launch the five runs
- [ ] Evaluation suite for answer and forecast metrics, K-sample spread baseline, sandbagging check
- [ ] Final results tables and figures; release checkpoints and data

## References

- [Beyond Binary Rewards: Training LMs to Reason About Their Uncertainty (RLCR)](https://arxiv.org/pdf/2507.16806): Brier-score calibration reward, BigMathDigits, test-time use of confidence
- [CALIBER: Calibrating Confidence Before and After Reasoning](https://arxiv.org/pdf/2606.24281): pre-thinking confidence against the group solve rate; the main baseline
- [Generalized Correctness Models](https://arxiv.org/pdf/2509.24988): little evidence of special self-knowledge of correctness; motivates the cross-model forecaster
- [Trained on Tokens, Calibrated on Concepts (ICLR 2026)](https://proceedings.iclr.cc/paper_files/paper/2026/file/39b228b1a83232b14a1c81b0ed5a7a05-Paper-Conference.pdf): models predicting their own answer distribution from the question alone
- [Calibrating Expressions of Certainty](https://arxiv.org/pdf/2410.04315): prompting models to state Beta parameters
- [Retrieval-Augmented Linguistic Calibration](https://arxiv.org/pdf/2605.19344): Beta distributions over reader interpretations
- Related work to locate in the literature sweep: COREA, CoCA and DCPO (cited by CALIBER); Laplace-LoRA (Yang et al., ICLR 2024); Ensemble Distribution Distillation (Malinin, Mlodozeniec and Gales); Bengs, Hüllermeier and Waegeman on second-order uncertainty from loss minimisation; Prior Networks and evidential deep learning
