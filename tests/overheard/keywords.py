"""Which keywords are listened for in a language: its own phrases plus the English base list,
the English ones both as English and as spoken with the language's own sounds."""
import json
from pathlib import Path
import wph

P = json.load(open(Path(__file__).parent / "phrases.json"))
LANGS = [k for k in P if not k.startswith("_")]

def keywords(lang):
    """[(phrase, variant, phones)]"""
    out = []
    own = P[lang]
    for ph in own["phrases"]:
        out.append((ph, own["espeak"], wph.phones(ph, own["espeak"])))
    if lang != "en":
        for ph in P["en"]["phrases"]:
            out.append((ph, "en-us", wph.phones(ph, "en-us")))
            try:
                out.append((ph, own["espeak"], wph.phones(ph, own["espeak"])))
            except ValueError:
                pass
    return out
