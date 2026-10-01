from pathlib import Path

import pytest

from conftest import REPO_ROOT, handled_message, say, utterance_message

DIALOGS = ("said_foul_language", "said_foul_word", "said_insult", "was_polite")
VOCABS = ("and", "foul_language", "polite_words")


def test_never_claims_an_utterance(plugin):
    for text in ("what time is it", "please tell me a joke", "what the fuck is this shit"):
        assert say(plugin, text) is None


def test_single_foul_word(plugin):
    say(plugin, "what the fuck")
    assert plugin.spoken == [("said_foul_word", {"foul_word": "fuck"})]


def test_several_foul_words_are_listed(plugin):
    say(plugin, "what the fuck is this shit")
    assert len(plugin.spoken) == 1
    name, data = plugin.spoken[0]
    assert name == "said_foul_language"
    assert " and " in data["foul_words"]


def test_whole_words_only(plugin):
    say(plugin, "the class assignment is on the fuchsia page")
    assert plugin.spoken == []


def test_nothing_is_said_before_the_utterance_is_handled(plugin):
    plugin.match(["what the fuck"], "en-US", utterance_message())
    assert plugin.spoken == []
    plugin.handle_utterance_handled(handled_message())
    assert plugin.spoken


def test_polite_streak_earns_a_compliment(plugin):
    say(plugin, "please tell me a joke")
    assert plugin.spoken == []
    say(plugin, "what is the weather please")
    assert plugin.spoken == [("was_polite", {})]


def test_foul_language_resets_the_streak(plugin):
    say(plugin, "please tell me a joke")
    say(plugin, "shit")
    plugin.spoken.clear()
    say(plugin, "please tell me the time")
    assert plugin.spoken == []


def test_sessions_are_independent(plugin):
    say(plugin, "please tell me a joke", session_id="a")
    say(plugin, "please tell me a joke", session_id="b")
    assert plugin.spoken == []
    plugin.match(["fuck"], "en-US", utterance_message("a"))
    plugin.handle_utterance_handled(handled_message("b"))
    assert plugin.spoken == []


def test_intent_probes_are_ignored(plugin):
    probe = utterance_message(msg_type="intent.service.intent.get")
    assert plugin.match(["what the fuck"], "en-US", probe) is None
    plugin.handle_utterance_handled(handled_message())
    assert plugin.spoken == []


def test_multilingual_matching_counts_once(plugin):
    msg = utterance_message()
    plugin.match(["please tell me a joke"], "en-US", msg)
    plugin.match(["please tell me a joke"], "de-DE", msg)
    plugin.handle_utterance_handled(handled_message())
    assert plugin.spoken == []  # threshold is 2, one utterance seen


def test_end_marker_of_another_utterance_is_ignored(plugin):
    plugin.match(["fuck"], "en-US", utterance_message(utterance_id="u1"))
    plugin.handle_utterance_handled(handled_message(utterance_id="u0"))
    assert plugin.spoken == []
    plugin.handle_utterance_handled(handled_message(utterance_id="u1"))
    assert plugin.spoken


def test_danish(plugin):
    say(plugin, "hvad fanden er klokken", lang="da-DK")
    assert plugin.spoken == [("said_foul_word", {"foul_word": "fanden"})]


@pytest.mark.parametrize("lang", sorted(p.name for p in (REPO_ROOT / "locale").iterdir()))
def test_every_language_has_all_resources(lang):
    folder = REPO_ROOT / "locale" / lang
    for name in DIALOGS:
        assert (folder / f"{name}.dialog").is_file(), f"{lang}/{name}.dialog"
    for name in VOCABS:
        assert (folder / f"{name}.voc").is_file(), f"{lang}/{name}.voc"


def test_spoken_on_the_bus_in_the_utterance_language():
    """Real speak(), no stubs: the comeback goes out on the bus, rendered in
    the language of the utterance and addressed to the requesting session,
    even when the end marker itself carries no language."""
    from ovos_utils.fakebus import FakeBus
    from conftest import GoodMannersPipeline

    bus = FakeBus()
    spoken = []
    bus.on("speak", lambda m: spoken.append(
        (m.data.get("utterance"), m.context.get("session", {}).get("session_id"))))
    plugin = GoodMannersPipeline(bus=bus)
    plugin.match(["hvad fanden er det for noget lort"], "da-DK", utterance_message("hive-1", "da-DK"))
    plugin.handle_utterance_handled(handled_message("hive-1"))
    assert len(spoken) == 1
    text, session_id = spoken[0]
    assert session_id == "hive-1"
    assert "lort og fanden" in text or "fanden og lort" in text
