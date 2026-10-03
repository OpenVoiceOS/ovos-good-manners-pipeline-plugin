#!/usr/bin/env python3
"""
Build locale/<lang>/foul_language.voc and slurs.voc from open word lists.

    python scripts/import_wordlists.py            # all languages
    python scripts/import_wordlists.py en-us      # one language

Two lists per language:

- foul_language.voc  ordinary swearing. The comeback repeats the word
                     ("shit is such an ugly word").
- slurs.voc          slurs and derogatory terms about groups of people.
                     The comeback does NOT repeat the word.

Sources (see README "Credits"):

- dsojevic/profanity-list (MIT, now randolf/profanity-list): the only
  English source, because it is tagged: racial / lgbtq, and religious at
  severity 3+, go to slurs; everything else to foul language. Clinical
  "-phile/-philia" terms are dropped; neutral words people use in normal
  questions ("sex", "porn", "nude", ...) are removed in the overrides.
- LDNOOBW (CC-BY-4.0): other languages. Untagged, so every language is
  sorted by hand in its override file.

Not used: wooorm/cuss and the English LDNOOBW list. ~1300 untagged
entries, many of them ethnic slurs (which would be said back),
ordinary words ("gay", "dumb", "butt") and porn search terms - LDNOOBW
started as a search filter - none of which belong in a voice assistant.

Hand-made corrections live in scripts/overrides/<lang>.json and always
win over the sources:

    {"foul": [...], "slurs": [...], "remove": [...]}

Words already in the .voc files are kept unless an override removes them,
so hand-written entries are never lost on a re-run. A word on both lists
ends up on slurs, the safer of the two (it is never said back).
"""

import json
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCALE = ROOT / "ovos_good_manners_pipeline_plugin" / "locale"
OVERRIDES = Path(__file__).resolve().parent / "overrides"

DSOJEVIC_EN = "https://raw.githubusercontent.com/randolf/profanity-list/main/en.json"
LDNOOBW = "https://raw.githubusercontent.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words/master/{code}"

# our locale folder -> LDNOOBW file name
LDNOOBW_CODES = {
    "da-dk": "da", "de-de": "de", "nl-nl": "nl", "sv-se": "sv",
    "nb-no": "no", "fr-fr": "fr", "es-es": "es", "it-it": "it", "pl-pl": "pl",
    "fi-fi": "fi", "pt-pt": "pt",
}

SLUR_TAGS = {"racial", "lgbtq"}


def fetch(url: str) -> str:
    with urllib.request.urlopen(url, timeout=30) as r:
        return r.read().decode("utf-8")


def normalize(phrase: str) -> str:
    return " ".join(phrase.lower().replace("*", "").split())


def read_voc(path: Path) -> set:
    if not path.is_file():
        return set()
    return {normalize(l) for l in path.read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.startswith("#")}


def write_voc(path: Path, words: set):
    path.write_text("".join(f"{w}\n" for w in sorted(words)), encoding="utf-8")


CLINICAL = re.compile(r"(phile|philia|philiac)$")


def english_sources():
    """(foul, slurs) from dsojevic."""
    foul, slurs = set(), set()
    for entry in json.loads(fetch(DSOJEVIC_EN)):
        # '*' means "previous character may repeat" - irrelevant for STT text
        variants = {normalize(v) for v in entry["match"].split("|") if v.strip()}
        variants = {v for v in variants if not CLINICAL.search(v)}
        tags = set(entry.get("tags", []))
        if tags & SLUR_TAGS or ("religious" in tags and entry["severity"] >= 3):
            slurs |= variants
        else:
            foul |= variants
    return foul, slurs


def build(lang: str):
    folder = LOCALE / lang
    foul = read_voc(folder / "foul_language.voc")
    slurs = read_voc(folder / "slurs.voc")

    code = LDNOOBW_CODES.get(lang)
    if code:
        foul |= {normalize(w) for w in fetch(LDNOOBW.format(code=code)).splitlines() if w.strip()}
    if lang == "en-us":
        en_foul, en_slurs = english_sources()
        foul |= en_foul
        slurs |= en_slurs

    override_file = OVERRIDES / f"{lang}.json"
    overrides = json.loads(override_file.read_text(encoding="utf-8")) if override_file.is_file() else {}
    foul |= {normalize(w) for w in overrides.get("foul", [])}
    slurs |= {normalize(w) for w in overrides.get("slurs", [])}
    remove = {normalize(w) for w in overrides.get("remove", [])}

    slurs -= remove
    foul -= remove | slurs
    foul = {w for w in foul if w}
    slurs = {w for w in slurs if w}

    folder.mkdir(parents=True, exist_ok=True)
    write_voc(folder / "foul_language.voc", foul)
    write_voc(folder / "slurs.voc", slurs)
    print(f"{lang}: {len(foul)} foul, {len(slurs)} slurs")


if __name__ == "__main__":
    langs = sys.argv[1:] or sorted(p.name for p in LOCALE.iterdir() if p.is_dir())
    for lang in langs:
        build(lang)
