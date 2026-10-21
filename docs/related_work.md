# Related work (Week 0 literature sweep)

Sweep date: 2026-10-21. Sources: web search, arXiv abstract pages, and the Semantic Scholar
citation API (133 papers citing RLCR, 3 citing CoCA, 1 citing CALIBER). Google Scholar and
OpenReview were reached only through web search, not queried directly; re-check both before
writing (PLAN.md risk table).

## Verdict

**The novelty claim still holds, but only in its narrow form.** No paper found trains a model to
verbalise a *distribution* over its own solve rate p_x, such as `m ± s` or Beta parameters, and
score it against k successes out of n rollouts. Every RL-trained pre-answer method we found
verbalises a point confidence and regresses it onto the group success rate. The closest threat
is **CoCA** (arXiv 2603.05881, March 2026). It puts a confidence *before* the answer, defines it
as the probability of solving the question under the current policy, rewards it against the
group-wise empirical success rate, and gives the confidence and answer segments separate
group-relative advantages. That is our Stage 1/2 design minus the spread and the Beta-Binomial
score. CoCA also predates CALIBER, so CALIBER is not the first pre-thinking group-target method.
Two of our planned contributions are therefore not new and should be presented as reused
machinery: (a) routing separate rewards/advantages to the forecast and answer spans (CoCA, DCPO,
Blockwise Advantage Estimation, MRTPO); (b) the worry that calibration rewards can make the model
fail on purpose (formalised as "confidence reward hacking" in arXiv 2607.04332). On the
second-order side, the nearest papers are prompt-only or post-hoc. One elicits interval-valued
"higher-order" uncertainty by prompting (2603.10396). One prompts for Beta parameters of
certainty phrases (2410.04315). The Beta-Bernoulli Calibrator (2605.27668) maps a point forecast
to a Beta over an *external event*, using human-forecaster disagreement for the spread, not the
model's own k-of-n. None of them trains on k-of-n outcomes, and none forecasts the model's own
capability. A reviewer will cite Bengs, Hüllermeier and Waegeman: no loss function makes a
second-order learner report epistemic uncertainty faithfully when it sees one label per input.
Our setup sidesteps this because each question has n > 1 rollouts, so Binomial overdispersion
identifies the Beta spread. We should say so explicitly, and describe `s` as the forecaster's
uncertainty about p_x, not as "epistemic uncertainty" in their sense. Arxiv 2609.09245 sets the
limit: n rollouts identify only the first n moments of the per-question success distribution.
**Safe claim:** the first method that trains a verbalised second-order self-forecast (a Beta
over one's own solve rate), scored as a Beta-Binomial prediction of k/n, with evidence that the
spread carries information beyond predicted difficulty.

## Most relevant papers

Threat = how much the paper undercuts the novelty claim. All links were checked on 2026-10-21:
arXiv IDs against the title on their abstract page, and the other URLs returned HTTP 200.

| # | Paper | Year | Relevance | Threat |
|---|---|---|---|---|
| 1 | [Confidence Before Answering (CoCA)](https://arxiv.org/abs/2603.05881) | 2026 | Pre-answer confidence = P(solve) under current policy; reward vs group empirical success rate; segmented advantages for confidence/answer. Our design minus spread | **high** (for pre-answer + routing); none for second-order |
| 2 | [CALIBER](https://arxiv.org/abs/2606.24281) | 2026 | Pre-thinking confidence vs group solve rate, post-answer vs correctness; main baseline | medium |
| 3 | [DCPO: Decoupling Reasoning and Confidence](https://arxiv.org/abs/2603.09117) | 2026 | Post-answer confidence with independent rewards, advantages and masked gradients per span; target blends group-mean and per-instance correctness | medium (routing) |
| 4 | [On the Pitfalls of Verbalized Confidence Priors (CalibSFT)](https://arxiv.org/abs/2609.32470) | 2026 | Verbal confidence collapses onto a few values during RL; an SFT stage with targets mixing success rate and correctness fixes it | low–medium |
| 5 | [Effectiveness of reward functions for RL confidence calibration](https://arxiv.org/abs/2607.04332) | 2026 | Defines "confidence reward hacking" (answering wrong to stay calibrated) and non-hackable reward schemes. Prior art for our sandbagging concern | medium (sandbagging framing) |
| 6 | [Blockwise Advantage Estimation](https://arxiv.org/abs/2602.10231) | 2026 | GRPO-compatible per-block advantages for multi-objective RL; outcome-conditioned baseline for later blocks | medium (routing) |
| 7 | [MRTPO: Each Reward to Its Relevant Tokens](https://arxiv.org/abs/2610.21439) | 2026 | Learned per-token routing of each reward's advantage | low |
| 8 | [Verbalizing Higher-order Uncertainty via Imprecise Probabilities](https://arxiv.org/abs/2603.10396) | 2026 | Prompt-elicited second-order uncertainty as probability intervals; no training, not k-of-n | low–medium |
| 9 | [Beta-Bernoulli Calibrator](https://arxiv.org/abs/2605.27668) | 2026 | Learned Beta(α,β) over event probability for LLM forecasts; variance = epistemic; trained on outcomes + human forecasts | low–medium |
| 10 | [Calibrating Expressions of Certainty](https://arxiv.org/abs/2410.04315) | 2024 | Certainty phrases as Beta distributions; prompts LMs for Beta parameters; post-hoc | low |
| 11 | [On Calibration of LLMs: From Response To Capability](https://arxiv.org/abs/2602.13540) | 2026 | "Capability calibration": query-level confidence vs expected accuracy p_x; evaluation framework, point estimate only | low (adopt the metric) |
| 12 | [Capability Self-Assessment in LLMs](https://arxiv.org/abs/2606.00251) | 2026 | SFT vs RL for self-assessment; notes training self-assessment changes the capability being assessed | low (self-fulfilling evidence) |
| 13 | [What Fixed-Rollout pass@k Evaluations Can Identify](https://arxiv.org/abs/2609.09245) | 2026 | n rollouts identify only n moments of the latent success distribution; Beta-Binomial forecasts depend on the mixing model | none (constrains our claims) |
| 14 | [LLMs Encode Their Failures (pre-generation probes)](https://arxiv.org/abs/2602.09924) · [code](https://github.com/KabakaWilliam/llms_know_difficulty) | 2026 | Linear probes on pre-generation activations predict policy-specific success; beats length/TF-IDF | low (Stage 0 baseline) |
| 15 | [Masked by Consensus](https://arxiv.org/abs/2604.12373) | 2026 | Self-probes ≈ peer-model probes for correctness; privileged knowledge only where models disagree | none (motivates cross-model forecaster) |
| 16 | [Generalized Correctness Models](https://arxiv.org/abs/2509.24988) | 2025 | Little special self-knowledge; history-based correctness predictors | none |
| 17 | [Future Confidence Distillation](https://arxiv.org/abs/2607.07626) | 2026 | Pre-solution hidden-state predictors distilled from post-solution probe confidence | low (Stage 0 baseline) |
| 18 | [Knowing Before Saying](https://arxiv.org/abs/2505.24362) | 2025 | Probe predicts CoT success before any token is generated | none |
| 19 | [Language Models (Mostly) Know What They Know (P(IK))](https://arxiv.org/abs/2207.05221) | 2022 | Original P(IK) head predicting own success from the question | none |
| 20 | [RLCR: Beyond Binary Rewards](https://arxiv.org/abs/2507.16806) | 2025 | Post-answer confidence with Brier reward; our post-answer component | low |
| 21 | [Uncalibrated Reasoning: GRPO Induces Overconfidence](https://arxiv.org/abs/2508.11800) | 2025 | GRPO std-normalisation miscalibrates probability outputs; removing it fixes it | none (RL design input) |
| 22 | [Beyond Verbalized Confidence (CREDO)](https://arxiv.org/abs/2609.34857) | 2026 | Sampled confidences collapse to a handful of values; replaces with differentiable token-pair readout | low |
| 23 | [Trained on Tokens, Calibrated on Concepts](https://arxiv.org/abs/2511.04869) | 2025 | Base LMs semantically calibrated; RL/CoT breaks it (ICLR 2026) | none |
| 24 | [Pitfalls of Epistemic UQ through Loss Minimisation](https://arxiv.org/abs/2203.06102) · [Second-Order Scoring Rules](https://arxiv.org/abs/2301.12736) | 2022–23 | No loss incentivises faithful second-order (epistemic) reports from single-label data | none (must address in framing) |
| 25 | [Prior Networks](https://arxiv.org/abs/1802.10501) · [Evidential DL](https://arxiv.org/abs/1806.01768) · [Ensemble Distribution Distillation](https://arxiv.org/abs/1905.00076) · [Laplace-LoRA](https://arxiv.org/abs/2308.13111) | 2018–23 | Dirichlet/Beta-output heads and distilled ensemble spread; our (α, β) head is a 2-class Prior Network trained on overdispersed counts | none |

Also checked and judged peripheral: [Let the Model Distribute Its Doubt](https://arxiv.org/abs/2511.14275)
(a distribution over *candidate answers*, so first-order), [Rewarding Doubt](https://arxiv.org/abs/2503.02623)
(log-score post-answer RL), [Looking Inward](https://arxiv.org/abs/2410.13787) (self-prediction of
behaviour), [GDPO](https://arxiv.org/abs/2601.05242) (per-reward normalisation),
[Agent failure prediction](https://arxiv.org/abs/2610.18614) (step confidence vs group episode
outcome), [DAR](https://arxiv.org/abs/2605.20740) (distribution from multiple sampled predictions,
not verbalised), [Test-time Calibration Learning](https://arxiv.org/abs/2610.02695) (the only
CALIBER citer on Semantic Scholar), and [COREA](https://arxiv.org/abs/2603.03752) (post-answer
confidence reward for cascades; cited by CALIBER).

**Unverified:**
- A web-search summary said CalibSFT (2609.32470) cites CALIBER. Its v1 HTML does not contain
  "CALIBER" and Semantic Scholar does not list the link, so treat it as unconfirmed.
- CoCA's exact confidence reward (Brier vs absolute error to the group success rate) comes from
  secondary summaries. Read the paper before re-implementing it.

## Suggested additions to PLAN.md

1. **Add CoCA as a baseline next to CALIBER** (decision 15). CoCA is the most direct pre-answer
   point-confidence competitor; CALIBER itself re-implements a "CoCA-lite". Both share our
   template, so the extra cost is one Stage 1 run.
2. **Rewrite the routing contribution as prior art.** Present the per-span routed run as CoCA/DCPO-style
   segmented advantages (cite Blockwise Advantage Estimation for the theory). Make the
   summed-vs-routed comparison and the sandbagging check the contribution, not routing itself.
3. **Check the summed reward against the non-hackable criterion of 2607.04332.** Run their check
   on the bounded RPS forecast reward plus correctness: can lowering p_x on purpose ever raise
   the expected total reward? Report the result next to the empirical sandbagging check.
4. **Watch for forecast collapse in RL.** CalibSFT and CREDO both report verbal confidences
   collapsing to a few values. Our SFT warm start should give broad support over (m, s), with
   targets balanced across the m range. Log the number of distinct (m, s) values per step
   alongside within-group spread.
5. **Do not std-normalise the forecast-span advantage** (2508.11800). GRPO std normalisation
   biases probabilistic outputs. Use mean-only baselines (RLOO/Dr. GRPO-style) for the forecast
   span, or add this as an ablation.
6. **Stage 0 baselines:**
   - The PIKA linear probe on pre-generation activations, with its length and TF-IDF surface
     baselines.
   - A peer-model probe (Masked by Consensus) to separate self-knowledge from shared difficulty.
   - Optionally, a Future-Confidence-Distillation-style head trained on post-solution probe
     targets.
7. **Report capability calibration** (2602.13540) as the named metric for m versus p_x, so the
   point-forecast part is comparable with prior work.
8. **Add an identifiability note to the evaluation.** With n rollouts, only n moments of p_x are
   identified (2609.09245), so claims about the spread `s` should be limited to what the
   Beta-Binomial on n samples tests: PIT on k, coverage, overdispersion. Consider a held-out
   subset with larger n (for example 32) to check `s` against the empirical spread of
   rollout-subsample solve rates.
9. **Framing paragraph on Bengs et al.** Explain why their impossibility result does not apply
   when there are n > 1 rollouts per input. Call `s` second-order uncertainty about one's own
   solve rate, and avoid the word "epistemic".
