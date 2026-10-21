import json

import pytest

from soc.data import TEMPLATE_HASH, messages
from soc.data import splits as S
from soc.data.sources import Question


def fake_pools():
    def make(prefix, n, kind="math"):
        return [
            Question(
                qid=f"{prefix}-{i:05d}",
                dataset=prefix,
                kind=kind,
                problem=f"{prefix} {i}",
                gold=str(i),
            )
            for i in range(n)
        ]

    test = make("bmd", 1000)
    # Train pool overlaps the test set on its first 50 questions and has duplicates.
    train = test[:50] + make("bmdtrain", 6500) + make("bmdtrain", 10)
    return {
        "bigmath_train_pool": train,
        "bigmath_test": test,
        "gpqa_diamond": make("gpqa", 198, "mcq"),
        "triviaqa_val": make("tqa", 3000, "short"),
        "simpleqa": make("sqa", 4326, "short"),
    }


@pytest.fixture
def frozen(tmp_path):
    pools = fake_pools()
    splits = S.build_all(load=pools.__getitem__)
    S.freeze(splits, tmp_path)
    return tmp_path, pools, splits


def test_sizes_and_disjointness(frozen):
    _, _, splits = frozen
    sizes = {k: len(v) for k, v in splits.items()}
    assert sizes == {
        "bigmath_train": 5000,
        "bigmath_val": 1000,
        "bigmath_test": 1000,
        "gpqa_diamond": 198,
        "triviaqa": 1000,
        "simpleqa": 1000,
    }
    train, val, test = (
        {q.qid for q in splits[k]} for k in ["bigmath_train", "bigmath_val", "bigmath_test"]
    )
    assert not (train & val) and not (train & test) and not (val & test)
    assert len(train) == 5000  # no duplicates


def test_selection_is_deterministic():
    a = S.build_all(load=fake_pools().__getitem__)
    b = S.build_all(load=fake_pools().__getitem__)
    assert all(S.split_hash(a[k]) == S.split_hash(b[k]) for k in a)


def test_selection_independent_of_upstream_order():
    pools = fake_pools()
    reversed_pools = {k: list(reversed(v)) for k, v in pools.items()}
    a = S.build_all(load=pools.__getitem__)
    b = S.build_all(load=reversed_pools.__getitem__)
    assert all(S.split_hash(a[k]) == S.split_hash(b[k]) for k in a)


def test_load_round_trip_and_tamper_detection(frozen):
    path, pools, splits = frozen
    loaded = S.load_split("bigmath_val", path, load=pools.__getitem__)
    assert [q.qid for q in loaded] == [q.qid for q in splits["bigmath_val"]]
    record = json.loads((path / "gpqa_diamond.json").read_text())
    assert "problem" not in json.dumps(record) and "gpqa 0" not in json.dumps(record)

    victim = splits["triviaqa"][0].qid
    edited = dict(pools)
    edited["triviaqa_val"] = [
        Question(qid=x.qid, dataset=x.dataset, kind=x.kind, problem=x.problem + "!", gold=x.gold)
        if x.qid == victim
        else x
        for x in pools["triviaqa_val"]
    ]
    with pytest.raises(ValueError, match=victim):
        S.load_split("triviaqa", path, load=edited.__getitem__)


def test_prompt_messages():
    msgs = messages(Question(qid="x", dataset="d", kind="mcq", problem="Pick one.", gold="A"))
    assert msgs[0]["role"] == "system" and "<answer>" in msgs[0]["content"]
    assert msgs[1]["content"].startswith("Pick one.") and "letter" in msgs[1]["content"]
    assert len(TEMPLATE_HASH) == 16


def test_clean_pool_drops_conflicts_and_dedupes():
    from soc.data.sources import clean_pool

    def mk(qid, gold):
        return Question(qid=qid, dataset="d", kind="math", problem=f"p{qid}", gold=gold)

    pool = [mk("a", "1"), mk("b", "2"), mk("a", "1"), mk("c", "3"), mk("b", "5")]
    assert [x.qid for x in clean_pool(pool)] == ["a", "c"]
