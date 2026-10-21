# second-order-conf

Can a language model state a calibrated, useful *spread* around its own solve rate?

Before reasoning the model writes `<forecast> m ± s </forecast>`, a belief about the probability
it solves the question. (m, s) becomes a Beta distribution, and its Beta-Binomial prediction of
k successes in n rollouts is scored with a proper score. See [PLAN.md](PLAN.md).

```bash
uv sync
uv run pytest
```

Licensed under Apache 2.0.
