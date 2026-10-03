# Copyright 2018, JarbasAI
# Copyright 2026, OpenVoiceOS
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
OVOS Good Manners - pipeline plugin

Originally a Mycroft skill by JarbasAI (2018), rewritten as an OVOS
pipeline plugin.

Reinforces good manners and reprimands foul language, without ever
getting in the way of what the user actually asked for.

How it works:

- The plugin sits FIRST in the intent pipeline, so it sees every
  utterance before any other stage can claim it.
- match() only observes: it classifies the utterance (polite / insult /
  foul language), stores the verdict for the session, and always returns
  None, so the utterance falls through to the stage that really handles it.
- When that utterance is finished (ovos.utterance.handled), the stored
  verdict is turned into a comeback: "you have really good manners",
  "X and Y are such ugly words", ...

The pipeline is selected per session (session.pipeline), so on a
HiveMind hub each client can have it enabled or not. Everything is kept
per session id for the same reason.

Classification is plain .voc matching for now. A trained
politeness/insult classifier can plug in at _classify() later; whatever
runs there runs on every single utterance, so it has to be fast.
"""

import re
import threading
import time
from dataclasses import dataclass, field
from os.path import dirname
from typing import Dict, List, Optional, Union

from ovos_bus_client.client import MessageBusClient
from ovos_bus_client.message import Message
from ovos_bus_client.session import SessionManager
from ovos_plugin_manager.templates.pipeline import IntentHandlerMatch, PipelinePlugin
from ovos_utils.fakebus import FakeBus
from ovos_utils.log import LOG
from ovos_workshop.app import OVOSAbstractApplication

PIPELINE_ID = "ovos-good-manners-pipeline-plugin"

# ovos-core also runs the pipeline for read-only intent probes
# (intent.service.intent.get). Those are not something the user said to
# the assistant, so they are neither counted nor answered.
PROBE_TOPICS = ("intent.service.intent.get",)

# end-of-utterance marker; on newer ovos-core it is guaranteed exactly once
# per utterance and carries an utterance_id
UTTERANCE_HANDLED = "ovos.utterance.handled"

DEFAULT_CONFIG = {
    # this many polite requests in a row earn a compliment
    "polite_threshold": 4,
    # minutes without a polite request before the count starts over
    "polite_timeout": 10,
    # seconds after the utterance after which a comeback is no longer said:
    # when ovos-core is busy (e.g. late common query answers on the
    # synchronous stable core), the end marker can arrive so late that the
    # comeback would land on top of the user's next request
    "max_delay": 20,
}

DEFAULT_SESSION_ID = "default"


@dataclass
class Verdict:
    """What _classify() found in one utterance."""
    polite: bool = False
    insult: bool = False
    slur: bool = False  # never said back, unlike foul_words
    foul_words: List[str] = field(default_factory=list)

    @property
    def rude(self) -> bool:
        return self.insult or self.slur or bool(self.foul_words)


@dataclass
class _Pending:
    """A verdict waiting for its utterance to be handled."""
    utterance_id: Optional[str]
    comebacks: List[str]
    foul_words: List[str]
    lang: str
    observed_at: float = 0.0


@dataclass
class _PoliteStreak:
    count: int = 0
    last: float = 0.0


def _session_id(message: Optional[Message]) -> str:
    if message is None:
        return DEFAULT_SESSION_ID
    try:
        return SessionManager.get(message).session_id or DEFAULT_SESSION_ID
    except Exception:  # malformed session carrier; nothing better to key on
        return DEFAULT_SESSION_ID


def _phrase_pattern(phrases: List[str]) -> Optional["re.Pattern"]:
    """One regex for a whole vocabulary: whole-word, case-insensitive.
    Lookarounds instead of \\b so phrases that start or end with symbols
    ("@$$") match as well. Longest first, so "motherfucker" wins over
    "fucker"."""
    phrases = sorted({p.strip().lower() for p in phrases if p.strip()}, key=len, reverse=True)
    if not phrases:
        return None
    return re.compile(r"(?<!\w)(?:" + "|".join(map(re.escape, phrases)) + r")(?!\w)")


def _prefix_pattern(prefixes: List[str]) -> Optional["re.Pattern"]:
    """Whole words that START with one of the prefixes. Danish, German and
    other compounding languages build swear words on the fly ("lortebil",
    "pissekoldt", "Scheißwetter"), which no word list can enumerate."""
    prefixes = sorted({p.strip().lower() for p in prefixes if p.strip()}, key=len, reverse=True)
    if not prefixes:
        return None
    return re.compile(r"(?<!\w)(?:" + "|".join(map(re.escape, prefixes)) + r")\w*")


class GoodMannersPipeline(PipelinePlugin, OVOSAbstractApplication):

    def __init__(self, bus: Optional[Union[MessageBusClient, FakeBus]] = None,
                 config: Optional[Dict] = None):
        OVOSAbstractApplication.__init__(
            self, bus=bus, skill_id=f"{PIPELINE_ID}.openvoiceos",
            resources_dir=dirname(__file__))
        PipelinePlugin.__init__(self, bus, config)
        self._init_state()
        self.add_event(UTTERANCE_HANDLED, self.handle_utterance_handled)

    def _init_state(self):
        self._lock = threading.Lock()
        self._pending: Dict[str, _Pending] = {}  # session id -> waiting verdict
        self._streaks: Dict[str, _PoliteStreak] = {}  # session id -> polite streak
        # session id -> key of the last utterance observed, so the same
        # utterance is only counted once when ovos-core calls match() for
        # several languages (intents.multilingual_matching)
        self._last_seen: Dict[str, object] = {}
        self._voc_cache_by_lang: Dict[tuple, List[str]] = {}
        self._pattern_cache: Dict[tuple, Optional[re.Pattern]] = {}

    def _setting(self, key):
        return (self.config or {}).get(key, DEFAULT_CONFIG[key])

    # --- pipeline ---------------------------------------------------------

    def match(self, utterances: List[str], lang: str,
              message: Message) -> Optional[IntentHandlerMatch]:
        """Observe the utterance, never claim it."""
        if message is None or message.msg_type in PROBE_TOPICS or not utterances:
            return None
        try:
            self._observe(utterances[0], lang, message)
        except Exception as e:  # never let this plugin break intent matching
            LOG.error(f"good manners: failed to observe utterance: {e}")
        return None

    def _observe(self, utterance: str, lang: str, message: Message):
        session_id = _session_id(message)
        utterance_id = message.context.get("utterance_id")
        # newer ovos-core stamps every utterance with an id; on older cores
        # the Message object is the same across the per-language calls
        key = utterance_id or id(message)
        with self._lock:
            if self._last_seen.get(session_id) == key:
                return
            self._last_seen[session_id] = key

        verdict = self._classify(utterance, lang)
        comebacks = self._comebacks_for(session_id, verdict)
        with self._lock:
            if comebacks:
                self._pending[session_id] = _Pending(
                    utterance_id, comebacks, verdict.foul_words, lang,
                    observed_at=time.monotonic())
            else:
                # a new utterance supersedes a comeback still waiting
                self._pending.pop(session_id, None)

    def _classify(self, utterance: str, lang: str) -> Verdict:
        """Plain vocabulary matching. This is where a trained classifier
        goes later; vocabulary matching stays as the fallback for
        languages the classifier does not cover."""
        text = utterance.lower()
        slur = self._pattern("slurs", lang)
        foul = self._pattern("foul_language", lang)
        polite = self._pattern("polite_words", lang)
        is_slur = bool(slur and slur.search(text))
        prefixed = self._pattern("foul_prefixes", lang, _prefix_pattern)
        found = (foul.findall(text) if foul else []) + (prefixed.findall(text) if prefixed else [])
        foul_words = list(dict.fromkeys(found))
        is_polite = bool(polite and polite.search(text))
        rude = is_slur or bool(foul_words)
        return Verdict(polite=is_polite and not rude, insult=False,
                       slur=is_slur, foul_words=foul_words)

    def _pattern(self, name: str, lang: str, build=None) -> Optional[re.Pattern]:
        key = (lang.lower(), name)
        if key not in self._pattern_cache:
            self._pattern_cache[key] = (build or _phrase_pattern)(self._vocabulary(name, lang))
        return self._pattern_cache[key]

    def _vocabulary(self, name: str, lang: str) -> List[str]:
        """The phrases of locale/<lang>/<name>.voc. Loaded through
        load_lang(lang) rather than voc_list(): voc_list() looks files up in
        the language of the message in flight, whatever lang it is given."""
        key = (lang.lower(), name)
        if key not in self._voc_cache_by_lang:
            try:
                lines = self.load_lang(lang=lang).load_vocabulary_file(name) or []
                phrases = [p for line in lines for p in line]
            except Exception:  # no vocabulary for this language
                phrases = []
            self._voc_cache_by_lang[key] = phrases
        return self._voc_cache_by_lang[key]

    def _speak_dialog_in(self, lang: str, name: str, data: Optional[dict] = None):
        """speak_dialog() in an explicit language: the comeback is spoken in
        the language the utterance was classified in."""
        data = data or {}
        try:
            utterance = self.load_lang(lang=lang).dialog_renderer.render(name, data)
        except Exception as e:
            LOG.error(f"good manners: no '{name}' dialog for {lang}: {e}")
            return
        self.speak(utterance, meta={"dialog": name, "data": data})

    def _comebacks_for(self, session_id: str, verdict: Verdict) -> List[str]:
        """Update the session's polite streak and decide what to say."""
        now = time.monotonic()
        with self._lock:
            streak = self._streaks.setdefault(session_id, _PoliteStreak())
            if verdict.rude:
                streak.count = 0
                # one comeback per utterance; a slur is the most serious and
                # is answered without repeating anything that was said
                if verdict.slur:
                    return ["said_slur"]
                if verdict.insult:
                    return ["said_insult"]
                return ["said_foul_language"]
            if not verdict.polite:
                return []
            if now - streak.last > self._setting("polite_timeout") * 60:
                streak.count = 0
            streak.count += 1
            streak.last = now
            if streak.count >= self._setting("polite_threshold"):
                streak.count = 0
                return ["was_polite"]
            return []

    # --- after the utterance ------------------------------------------------

    def handle_utterance_handled(self, message: Message):
        session_id = _session_id(message)
        with self._lock:
            # the next utterance is a new one even if its Message object
            # happens to get the same id() as this one
            self._last_seen.pop(session_id, None)
            pending = self._pending.get(session_id)
            if pending is None:
                return
            utterance_id = message.context.get("utterance_id")
            if pending.utterance_id and utterance_id and pending.utterance_id != utterance_id:
                return  # end marker of a different utterance
            del self._pending[session_id]
        age = time.monotonic() - pending.observed_at
        if age > self._setting("max_delay"):
            LOG.debug(f"good manners: dropping comeback, utterance was {age:.0f}s ago")
            return
        for comeback in pending.comebacks:
            if comeback == "said_foul_language":
                self._speak_foul_words(pending.foul_words, pending.lang)
            else:
                self._speak_dialog_in(pending.lang, comeback)

    def _speak_foul_words(self, words: List[str], lang: str):
        if len(words) == 1:
            self._speak_dialog_in(lang, "said_foul_word", {"foul_word": words[0]})
            return
        joiner = (self._vocabulary("and", lang) or ["and"])[0]
        listed = f"{', '.join(words[:-1])} {joiner} {words[-1]}"
        self._speak_dialog_in(lang, "said_foul_language", {"foul_words": listed})
