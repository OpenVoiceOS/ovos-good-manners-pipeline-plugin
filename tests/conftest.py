"""Shared fixtures for the good manners pipeline plugin tests."""
import sys
from pathlib import Path

import pytest
from ovos_bus_client.message import Message
from ovos_utils.fakebus import FakeBus

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "ovos_good_manners_pipeline_plugin"
sys.path.insert(0, str(REPO_ROOT))

from ovos_good_manners_pipeline_plugin import opm as module  # noqa: E402
sys.modules["good_manners_pipeline"] = module

GoodMannersPipeline = module.GoodMannersPipeline


def utterance_message(session_id="default", lang="en-US", utterance_id=None,
                      msg_type="recognizer_loop:utterance"):
    context = {"session": {"session_id": session_id, "lang": lang}}
    if utterance_id:
        context["utterance_id"] = utterance_id
    return Message(msg_type, {"utterances": ["..."], "lang": lang}, context)


def handled_message(session_id="default", utterance_id=None):
    context = {"session": {"session_id": session_id}}
    if utterance_id:
        context["utterance_id"] = utterance_id
    return Message("ovos.utterance.handled", {}, context)


@pytest.fixture
def plugin():
    p = GoodMannersPipeline(bus=FakeBus(), config={"polite_threshold": 2})
    p.spoken = []
    p._speak_dialog_in = lambda lang, name, data=None: p.spoken.append((name, data or {}))
    return p


def say(plugin, text, session_id="default", lang="en-US", utterance_id=None):
    """One full turn: the pipeline sees the utterance, then it is handled."""
    msg = utterance_message(session_id, lang, utterance_id)
    result = plugin.match([text], lang, msg)
    plugin.handle_utterance_handled(handled_message(session_id, utterance_id))
    return result
