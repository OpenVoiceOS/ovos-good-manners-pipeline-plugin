"""FLEURS test speech, extracted to one wav per utterance by extract_fleurs.py."""
import soundfile as sf
FLEURS = {"da": "da_dk", "en": "en_us", "de": "de_de", "sv": "sv_se", "nb": "nb_no",
          "nl": "nl_nl", "fr": "fr_fr", "es": "es_419", "it": "it_it"}
import os
DATA = os.environ.get("OVERHEARD_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
ROOT = os.path.join(DATA, "fleurs") + "/"

def utterances(lang, last=None):
    """(audio, text) for each utterance, in order; `last` gives only the final n."""
    d = ROOT + FLEURS[lang]
    rows = [l.rstrip("\n").split("\t", 1) for l in open(f"{d}/texts.tsv")]
    if last:
        rows = rows[-last:]
    for i, text in rows:
        a, sr = sf.read(f"{d}/{i}.wav", dtype="float32")
        assert sr == 16000
        yield (a.mean(1) if a.ndim > 1 else a), text
