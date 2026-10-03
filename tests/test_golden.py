"""Golden corpus: one line per utterance in tests/golden/<lang>.jsonl, with
what the plugin should find in it.

    {"utterance": "what time is it", "expect": "none"}
    {"utterance": "could you turn off the lights", "expect": "polite"}
    {"utterance": "you are retarded", "expect": "slur"}
    {"utterance": "fuck this shit", "expect": "foul", "foul_words": ["fuck", "shit"]}

"none" rows matter as much as the rest: they are the ordinary requests and
look-alike words ("summa cum laude", "røverhistorie", "Schmuck") that must
never be reprimanded. Adding a language means adding a file, no code.
"""
import json
from pathlib import Path

import pytest

GOLDEN = Path(__file__).parent / "golden"


def _rows():
    for path in sorted(GOLDEN.glob("*.jsonl")):
        lang = path.stem
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip():
                row = json.loads(line)
                yield pytest.param(lang, row, id=f"{lang}:{n}:{row['utterance']}")


@pytest.fixture(scope="module")
def classifier():
    from ovos_utils.fakebus import FakeBus
    from conftest import GoodMannersPipeline
    return GoodMannersPipeline(bus=FakeBus())


@pytest.mark.parametrize("lang,row", list(_rows()))
def test_golden(classifier, lang, row):
    verdict = classifier._classify(row["utterance"], lang)
    expect = row["expect"]
    if expect == "none":
        assert not verdict.rude and not verdict.polite, verdict
    elif expect == "polite":
        assert verdict.polite and not verdict.rude, verdict
    elif expect == "slur":
        assert verdict.slur, verdict
    elif expect == "foul":
        assert not verdict.slur, verdict
        assert sorted(verdict.foul_words) == sorted(row["foul_words"]), verdict
    else:
        pytest.fail(f"unknown expect: {expect}")
