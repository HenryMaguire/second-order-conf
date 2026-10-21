import pytest

from soc.forecast import (
    Forecast,
    format_forecast,
    parse_confidence,
    parse_forecast,
    parse_forecast_body,
)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("<forecast> 0.62 ± 0.14 </forecast>", (0.62, 0.14)),
        ("<forecast>0.62±0.14</forecast>", (0.62, 0.14)),
        ("<forecast> 0.62 +/- 0.14 </forecast>", (0.62, 0.14)),
        ("<forecast> 0.62 +- 0.14 </forecast>", (0.62, 0.14)),
        ("<forecast> 1 ± 0 </forecast>", (1.0, 0.0)),
        ("<forecast> .5 ± .25 </forecast>", (0.5, 0.25)),
        ("<forecast>\n0.00 ± 0.50\n</forecast>\n<think>...</think>", (0.0, 0.5)),
        ("prefix <forecast> 0.3 ± 0.1 </forecast> suffix", (0.3, 0.1)),
    ],
)
def test_valid_forecasts(text, expected):
    assert parse_forecast(text) == Forecast(*expected)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "0.62 ± 0.14",
        "<forecast> 0.62 ± 0.14",
        "<forecast> 0.62 </forecast>",
        "<forecast> ± 0.14 </forecast>",
        "<forecast> 0.62 ± </forecast>",
        "<forecast> 62% ± 14% </forecast>",
        "<forecast> 1.2 ± 0.1 </forecast>",
        "<forecast> 0.5 ± 0.6 </forecast>",
        "<forecast> -0.1 ± 0.1 </forecast>",
        "<forecast> 0.5 ± -0.1 </forecast>",
        "<forecast> nan ± 0.1 </forecast>",
        "<forecast> inf ± 0.1 </forecast>",
        "<forecast> 1e-1 ± 0.1 </forecast>",
        "<forecast> 0.62 ± 0.14 because </forecast>",
        "<forecast> about 0.62 ± 0.14 </forecast>",
        "<forecast> 0.62 ± 0.14 ± 0.1 </forecast>",
        "<forecast> 0.6 ± 0.1 </forecast><forecast> 0.7 ± 0.1 </forecast>",
        "</forecast> 0.6 ± 0.1 <forecast>",
        "<forecast> 0,62 ± 0,14 </forecast>",
    ],
)
def test_malformed_forecasts(text):
    assert parse_forecast(text) is None


def test_body_parser_for_stage1_completions():
    # Stage 1 prompts end with "<forecast>" and generation stops at "</forecast>".
    assert parse_forecast_body(" 0.62 ± 0.14 ") == Forecast(0.62, 0.14)
    assert parse_forecast_body(" 0.62 ± 0.14 </forecast>") is None


def test_format_round_trip():
    body = format_forecast(0.6234, 0.1389)
    assert body == "0.62 ± 0.14"
    assert parse_forecast_body(body) == Forecast(0.62, 0.14)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("<confidence> 0.8 </confidence>", 0.8),
        ("<answer>4</answer><confidence>1</confidence>", 1.0),
        ("<confidence> 0 </confidence>", 0.0),
        ("<confidence> 1.5 </confidence>", None),
        ("<confidence> high </confidence>", None),
        ("<confidence> 0.8", None),
        ("<confidence>0.1</confidence><confidence>0.9</confidence>", None),
    ],
)
def test_confidence(text, expected):
    assert parse_confidence(text) == expected
