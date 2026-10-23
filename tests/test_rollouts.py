import pandas as pd
import pytest
import yaml

from soc.data.sources import Question
from soc.rollouts import run as R
from soc.rollouts.generate import Sample

QUESTIONS = {
    "bigmath_val": [
        Question(qid=f"bmd-{i}", dataset="bigmath", kind="math", problem=f"{i}+0?", gold=str(i))
        for i in range(7)
    ],
    "gpqa_diamond": [
        Question(qid=f"gpqa-{i}", dataset="gpqa", kind="mcq", problem="Pick.", gold="A")
        for i in range(3)
    ],
}


class FakeGenerator:
    """Question i gets i correct answers out of n (capped), so k is known in advance."""

    name = "fake"

    def __init__(self, fail_after: int | None = None):
        self.calls, self.fail_after, self.seeds = 0, fail_after, []

    def generate(self, conversations, n, temperature, max_new_tokens, seed):
        if self.fail_after is not None and self.calls >= self.fail_after:
            raise RuntimeError("pre-empted")
        self.calls += 1
        self.seeds.append(seed)
        out = []
        for conv in conversations:
            prompt = conv[1]["content"]
            if prompt.startswith("Pick"):
                out.append([Sample("<answer>A</answer>", 5, "stop")] * n)
                continue
            i = int(prompt.split("+")[0])
            right = [Sample(f"so <answer>{i}</answer>", 10, "stop")] * min(i, n)
            wrong = [Sample("no answer", max_new_tokens, "length")] * (n - min(i, n))
            out.append(right + wrong)
        return out


@pytest.fixture
def config(tmp_path, monkeypatch):
    cfg = {
        "run_id": "test",
        "model": "fake",
        "splits": ["bigmath_val", "gpqa_diamond"],
        "n": 32,
        "temperature": 0.7,
        "max_new_tokens": 2048,
        "shard_size": 3,
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setenv("SOC_OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.delenv("SOC_OUTPUT_URI", raising=False)
    return path


def test_counts_and_layout(config):
    out = R.main(str(config), generator=FakeGenerator(), load=QUESTIONS.__getitem__)
    counts = pd.read_parquet(out / "counts" / "bigmath_val.parquet")
    assert counts.qid.tolist() == [f"bmd-{i}" for i in range(7)]
    assert counts.k.tolist() == list(range(7)) and (counts.n == 32).all()
    assert counts.trunc_rate.tolist() == pytest.approx([(32 - i) / 32 for i in range(7)])
    assert (pd.read_parquet(out / "counts" / "gpqa_diamond.parquet").k == 32).all()
    assert len(list((out / "rollouts" / "bigmath_val").glob("shard-*.parquet"))) == 3
    rows = pd.read_parquet(out / "rollouts" / "bigmath_val" / "shard-00000.parquet")
    assert len(rows) == 3 * 32 and set(rows.columns) >= {"response", "answer", "correct"}
    assert (out / "run.json").exists()


def test_resume_after_preemption_skips_finished_shards(config):
    with pytest.raises(RuntimeError, match="pre-empted"):
        R.main(str(config), generator=FakeGenerator(fail_after=2), load=QUESTIONS.__getitem__)
    gen = FakeGenerator()
    out = R.main(str(config), generator=gen, load=QUESTIONS.__getitem__)
    # 3 bigmath shards + 1 gpqa shard in total; 2 were finished before the crash.
    assert gen.calls == 2
    assert pd.read_parquet(out / "counts" / "bigmath_val.parquet").k.tolist() == list(range(7))
    assert not list(out.rglob("*.tmp"))


def test_shard_seeds_differ_and_are_stable(config):
    a = FakeGenerator()
    R.main(str(config), generator=a, load=QUESTIONS.__getitem__)
    assert len(set(a.seeds)) == len(a.seeds)
    assert R._shard_seed(0, "bigmath_val", 1) == R._shard_seed(0, "bigmath_val", 1)


def test_refuses_settings_that_differ_from_frozen(config):
    cfg = yaml.safe_load(config.read_text())
    cfg["n"] = 64
    config.write_text(yaml.safe_dump(cfg))
    with pytest.raises(SystemExit, match="n=64"):
        R.main(str(config), generator=FakeGenerator(), load=QUESTIONS.__getitem__)


def test_shipped_config_matches_frozen():
    R.check_frozen(R.RolloutConfig.from_yaml(R.ROOT / "configs" / "stage0_rollouts.yaml"))


def test_judge_pass_adds_judged_counts(tmp_path, monkeypatch):
    from soc.rollouts.judge import judge_split
    from soc.verify.judge import Judge

    qs = [
        Question(qid="t0", dataset="tqa", kind="short", problem="Who?", gold="Michio Sugeno"),
        Question(qid="t1", dataset="tqa", kind="short", problem="Where?", gold="Paris"),
    ]

    class Gen:
        name = "fake"

        def generate(self, conversations, n, temperature, max_new_tokens, seed):
            outs = []
            for c in conversations:
                if c[1]["content"].startswith("Who"):
                    outs.append([Sample("<answer>M. Sugeno</answer>", 3, "stop")] * n)
                else:
                    outs.append(
                        [Sample("<answer>paris</answer>", 3, "stop")] * (n - 2)
                        + [Sample("<answer>Lyon</answer>", 3, "stop")]
                        + [Sample("no tags at all", 3, "stop")]  # missing answer -> NaN
                    )
            return outs

    cfg = {
        "run_id": "t",
        "model": "fake",
        "splits": ["triviaqa"],
        "n": 32,
        "temperature": 0.7,
        "max_new_tokens": 2048,
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setenv("SOC_OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.delenv("SOC_OUTPUT_URI", raising=False)
    load = {"triviaqa": qs}.__getitem__
    out = R.main(str(path), generator=Gen(), load=load)

    class Client:
        calls = 0
        messages = None

        def create(self, **kw):
            Client.calls += 1
            text = "CORRECT" if "M. Sugeno" in kw["messages"][0]["content"] else "INCORRECT"
            from types import SimpleNamespace as NS

            return NS(
                stop_reason="end_turn", stop_details=None, content=[NS(type="text", text=text)]
            )

    client = Client()
    client.messages = client
    counts = judge_split(out, "triviaqa", Judge(tmp_path / "j.sqlite", client=client), load=load)
    assert counts.k.tolist() == [0, 30]  # exact match only
    assert counts.k_judged.tolist() == [32, 30]
    assert Client.calls == 2  # one call per distinct non-matching answer


def test_max_shards_limits_work(config):
    cfg = yaml.safe_load(config.read_text())
    cfg["max_shards"] = 1
    config.write_text(yaml.safe_dump(cfg))
    gen = FakeGenerator()
    out = R.main(str(config), generator=gen, load=QUESTIONS.__getitem__)
    assert gen.calls == 2  # one shard per split
    counts = pd.read_parquet(out / "counts" / "bigmath_val.parquet")
    assert counts.qid.tolist() == ["bmd-0", "bmd-1", "bmd-2"]


def test_smoke_config_matches_frozen():
    R.check_frozen(R.RolloutConfig.from_yaml(R.ROOT / "configs" / "stage0_smoke.yaml"))


def test_batches_extend_the_same_run(config):
    gen = FakeGenerator()
    out = R.main(str(config), generator=gen, load=QUESTIONS.__getitem__, shards={"bigmath_val": 1})
    assert gen.calls == 1
    assert not (out / "counts" / "gpqa_diamond.parquet").exists()
    assert pd.read_parquet(out / "counts" / "bigmath_val.parquet").qid.tolist() == [
        "bmd-0",
        "bmd-1",
        "bmd-2",
    ]
    gen2 = FakeGenerator()
    R.main(
        str(config),
        generator=gen2,
        load=QUESTIONS.__getitem__,
        shards={"bigmath_val": 3, "gpqa_diamond": 1},
    )
    assert gen2.calls == 3  # bigmath shards 1 and 2, gpqa shard 0; shard 0 is reused
    counts = pd.read_parquet(out / "counts" / "bigmath_val.parquet")
    assert counts.k.tolist() == list(range(7))
    assert len((out / "invocations.jsonl").read_text().splitlines()) == 2


def test_parse_shards():
    assert R.parse_shards("bigmath_train=4, bigmath_val=2") == {
        "bigmath_train": 4,
        "bigmath_val": 2,
    }


def test_ambigqa_config_matches_frozen():
    R.check_frozen(R.RolloutConfig.from_yaml(R.ROOT / "configs" / "stage0_ambigqa.yaml"))
