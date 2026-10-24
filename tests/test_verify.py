from types import SimpleNamespace

import pytest

from soc.data.sources import Question
from soc.verify import extract_answer, verify, verify_with_judge
from soc.verify.checks import check_math, check_mcq, exact_match, to_number
from soc.verify.judge import Judge


@pytest.mark.parametrize(
    "text,expected",
    [
        ("reasoning <answer>42</answer>", "42"),
        ("<answer>1</answer> wait <answer> 2 </answer>", "2"),
        ("<answer>\n3.5\n</answer>", "3.5"),
        ("no tags", None),
        ("<answer>  </answer>", None),
        ("<answer>unterminated", None),
    ],
)
def test_extract_answer(text, expected):
    assert extract_answer(text) == expected


@pytest.mark.parametrize(
    "text,value",
    [
        ("42", 42),
        (" 2999999999999 ", 2999999999999),
        ("-20.5", -20.5),
        ("$12.5$", 12.5),
        ("\\boxed{10}", 10),
        ("x = 7", 7),
        ("1,234", 1234),
        ("1,234.5", 1234.5),
        ("3/4", 0.75),
        ("\\frac{3}{4}", 0.75),
        ("-\\dfrac{1}{2}", -0.5),
        ("10 cm", 10),
        ("25%", 25),
        ("6.", 6),
        ("0.000216", 0.000216),
    ],
)
def test_to_number(text, value):
    assert to_number(text) == pytest.approx(value)


@pytest.mark.parametrize("text", ["", "seven", "1, 2", "1/0", "1 or 2", "2x+1"])
def test_to_number_rejects(text):
    assert to_number(text) is None


def test_check_math():
    assert check_math("10", "10")
    assert check_math("10.0", "10")
    assert check_math("\\boxed{48.21}", "48.21")
    assert not check_math("48.2", "48.21")
    assert not check_math("11", "10")
    assert check_math("\\sqrt{4}", "2")  # math-verify fallback
    assert not check_math("banana", "2")


@pytest.mark.parametrize(
    "pred,ok",
    [
        ("D", True),
        ("d", True),
        ("(D)", True),
        ("D. a planet", True),
        ("The answer is D", True),
        ("Answer: D", True),
        ("C", False),
        ("Dense", False),
        ("", False),
    ],
)
def test_check_mcq(pred, ok):
    assert check_mcq(pred, "D") is ok


def test_exact_match():
    golds = ["David Seville", "Ross Bagdasarian"]
    assert exact_match("david seville", golds)
    assert exact_match("The Ross Bagdasarian.", golds)
    assert not exact_match("Seville", golds)
    assert not exact_match("", golds)


def q(kind, gold, aliases=()):
    return Question(qid="x", dataset="t", kind=kind, problem="Q?", gold=gold, aliases=aliases)


def test_verify_dispatch():
    assert verify(q("math", "10"), "so <answer>10</answer>")
    assert not verify(q("math", "10"), "so 10")
    assert verify(q("mcq", "B"), "<answer>B</answer>")
    assert verify(q("short", "Paris", ("paris", "Paris, France")), "<answer>Paris, France</answer>")


class FakeClient:
    def __init__(self, reply):
        self.reply, self.calls = reply, 0
        self.messages = self

    def create(self, **kwargs):
        self.calls += 1
        assert kwargs["model"] == "claude-haiku-5-5"
        return SimpleNamespace(
            stop_reason="end_turn",
            stop_details=None,
            content=[
                SimpleNamespace(type="thinking", thinking=""),
                SimpleNamespace(type="text", text=self.reply),
            ],
        )


def test_judge_caches(tmp_path):
    client = FakeClient("CORRECT")
    judge = Judge(tmp_path / "j.sqlite", client=client)
    assert judge.grade("Q?", ["Michio Sugeno"], "M. Sugeno") == "CORRECT"
    assert judge.grade("Q?", ["Michio Sugeno"], "M. Sugeno") == "CORRECT"
    assert client.calls == 1
    # Cache persists across instances.
    judge2 = Judge(tmp_path / "j.sqlite", client=FakeClient("INCORRECT"))
    assert judge2.grade("Q?", ["Michio Sugeno"], "M. Sugeno") == "CORRECT"


@pytest.mark.parametrize(
    "reply,verdict",
    [("INCORRECT", "INCORRECT"), ("NOT_ATTEMPTED", "NOT_ATTEMPTED"), ("Grade: correct", "CORRECT")],
)
def test_judge_parses(tmp_path, reply, verdict):
    assert Judge(tmp_path / "j.sqlite", client=FakeClient(reply)).grade("Q", ["a"], "b") == verdict


def test_judge_unparseable_raises(tmp_path):
    with pytest.raises(RuntimeError):
        Judge(tmp_path / "j.sqlite", client=FakeClient("maybe")).grade("Q", ["a"], "b")


def test_verify_with_judge_skips_call_on_exact_match(tmp_path):
    client = FakeClient("INCORRECT")
    judge = Judge(tmp_path / "j.sqlite", client=client)
    question = q("short", "Michio Sugeno")
    assert verify_with_judge(question, "<answer>michio sugeno</answer>", judge)
    assert client.calls == 0
    assert not verify_with_judge(question, "<answer>Someone else</answer>", judge)
    assert not verify_with_judge(question, "no answer tag", judge)
    assert client.calls == 1


def test_contains_match():
    from soc.verify.checks import contains_match

    assert contains_match("November 18, 2000", ["2000"])
    assert contains_match("Central Station (Hauptbahnhof)", ["Hauptbahnhof"])
    assert contains_match("Fascist", ["National Fascist Party"])  # answer inside gold
    assert not contains_match("20001", ["2000"])  # whole words only
    assert not contains_match("Mount Herbert", ["Ditchling Beacon"])
    assert not contains_match("", ["x"])
