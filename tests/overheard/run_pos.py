"""Positives and look-alikes: every phrase spoken by many Piper voices, in four conditions
(clean, room echo, noise, other people talking), scored with every keyword of the language.
-> scores/pos_<lang>.npz

Danish has only one Piper voice, so Norwegian and Swedish voices are also made to speak Danish
(Danish phonemes into their models): Danish with a Scandinavian accent, but different voices.
The English base phrases are also spoken by each language's own voices, the way a Dane says
"what the fuck"."""
import gc, io, json, random, sys
import numpy as np, pyarrow.parquet as pq, soundfile as sf
from piper import PiperVoice, SynthesisConfig
import wph
from fleurs import utterances
from keywords import keywords, P
from fleurs import FLEURS

from fleurs import DATA
VOICES = DATA + "/piper/"
# lang -> [(model, n speakers to sample (None = single), espeak override)]
CAST = {
    "en": [("en_US-libritts_r-medium", 6, None), ("en_GB-vctk-medium", 3, None), ("en_US-l2arctic-medium", 3, None),
           ("en_US-lessac-medium", None, None), ("en_GB-alan-medium", None, None), ("en_US-ryan-medium", None, None),
           ("en_US-amy-medium", None, None)],
    "da": [("da_DK-talesyntese-medium", None, None), ("no_NO-nvcc-medium", 6, "da"), ("no_NO-talesyntese-medium", None, "da"),
           ("sv_SE-nst-medium", None, "da"), ("sv_SE-lisa-medium", None, "da"), ("sv_SE-alma-medium", None, "da")],
    "de": [("de_DE-mls-medium", 6, None), ("de_DE-thorsten-medium", None, None), ("de_DE-kerstin-low", None, None),
           ("de_DE-eva_k-x_low", None, None), ("de_DE-karlsson-low", None, None), ("de_DE-pavoque-low", None, None),
           ("de_DE-ramona-low", None, None)],
    "sv": [("sv_SE-nst-medium", None, None), ("sv_SE-lisa-medium", None, None), ("sv_SE-alma-medium", None, None),
           ("no_NO-nvcc-medium", 6, "sv")],
    "nb": [("no_NO-nvcc-medium", 10, None), ("no_NO-talesyntese-medium", None, None)],
    "nl": [("nl_NL-mls-medium", 5, None), ("nl_NL-alex-medium", None, None), ("nl_NL-pim-medium", None, None),
           ("nl_NL-ronnie-medium", None, None), ("nl_BE-rdh-medium", None, None), ("nl_BE-nathalie-medium", None, None),
           ("nl_BE-flemishguy-medium", None, None)],
    "fr": [("fr_FR-mls-medium", 6, None), ("fr_FR-siwis-medium", None, None), ("fr_FR-tom-medium", None, None),
           ("fr_FR-gilles-low", None, None), ("fr_FR-upmc-medium", 2, None)],
    "es": [("es_ES-sharvard-medium", 2, None), ("es_ES-davefx-medium", None, None), ("es_ES-carlfm-x_low", None, None),
           ("es_AR-daniela-high", None, None), ("es_MX-ald-medium", None, None), ("es_MX-claude-high", None, None),
           ("es_ES-mls_9972-low", None, None), ("es_ES-mls_10246-low", None, None)],
    "it": [("it_IT-paola-medium", None, None), ("it_IT-riccardo-x_low", None, None), ("it_IT-serena-medium", None, None)],
}
CONDITIONS = ["clean", "room", "noise10", "babble10"]


def to16k(a, sr):
    if sr == 16000:
        return a.astype(np.float32)
    t = np.arange(int(len(a) * 16000 / sr)) * sr / 16000
    return np.interp(t, np.arange(len(a)), a).astype(np.float32)


def pink(n, rng):
    f = np.fft.rfftfreq(n); f[0] = f[1]
    x = np.fft.irfft((rng.normal(size=len(f)) + 1j * rng.normal(size=len(f))) / np.sqrt(f), n)
    return (x / (x.std() + 1e-9)).astype(np.float32)


def at_snr(sig, noise, snr):
    ps = np.mean(sig[np.abs(sig) > 0.01 * np.abs(sig).max()] ** 2)
    return sig + noise * np.sqrt(ps / (np.mean(noise ** 2) * 10 ** (snr / 10)))


def room(sig, rng):
    """A simple living-room echo: exponentially decaying noise, RT60 0.3-0.6 s."""
    rt = rng.uniform(0.3, 0.6); n = int(rt * 16000)
    ir = rng.normal(size=n) * np.exp(-6.9 * np.arange(n) / n); ir[0] = 1.0
    out = np.convolve(sig, ir)[:len(sig)]
    return (out / (np.abs(out).max() + 1e-9) * np.abs(sig).max()).astype(np.float32)


def condition(clip, cond, rng, babble):
    if cond == "clean":
        return clip
    if cond == "room":
        return room(clip, rng)
    if cond == "noise10":
        return at_snr(clip, pink(len(clip), rng), 10)
    b = babble[rng.integers(len(babble))]
    start = rng.integers(max(1, len(b) - len(clip)))
    seg = np.resize(b[start:start + len(clip)], len(clip))
    return at_snr(clip, seg, 10)


def main(lang):
    rng = np.random.default_rng(42); random.seed(42)
    kws = keywords(lang)
    sp = wph.Spotter()
    # babble: the last 30 FLEURS utterances (other people talking in the room)
    babble = [a for a, _ in utterances(lang, last=30)]
    own = P[lang]
    texts = [(ph, "pos", ph) for ph in own["phrases"]] + [(s, "look", None) for s in own["lookalikes"]]
    en_texts = [] if lang == "en" else [(ph, "pos", ph) for ph in P["en"]["phrases"]]
    rows = []
    sil = np.zeros(9600, np.float32)
    for model, nspk, override in CAST[lang]:
        voice = PiperVoice.load(VOICES + model + ".onnx")
        if override:
            voice.config.espeak_voice = override
        speakers = [None] if not nspk else rng.choice(voice.config.num_speakers, nspk, replace=False).tolist()
        for spk in speakers:
            mine = texts + en_texts
            for text, kind, target in mine:
                cfg = SynthesisConfig(speaker_id=spk, length_scale=float(rng.uniform(0.85, 1.25)))
                audio = np.concatenate([c.audio_float_array for c in voice.synthesize(text, cfg)])
                clip = np.concatenate([sil, to16k(audio, voice.config.sample_rate), sil])
                for cond in CONDITIONS:
                    x = condition(clip, cond, rng, babble)
                    x = (x / (np.abs(x).max() + 1e-9) * rng.uniform(0.2, 0.8)).astype(np.float32)
                    s, v = sp.stream(x, [k[2] for k in kws])
                    rows.append(dict(voice=f"{model}#{spk}" if spk is not None else model, accent=override or "",
                                     text=text, kind=kind, target=target or "", cond=cond,
                                     dur=len(audio) / voice.config.sample_rate, peak=s.max(0).astype(np.float32)))
        print(lang, model, len(rows), flush=True)
        del voice; gc.collect()
    np.savez_compressed(f"scores/pos_{lang}.npz", peak=np.stack([r.pop("peak") for r in rows]),
                        meta=np.array([json.dumps(r, ensure_ascii=False) for r in rows]),
                        kw_phrase=np.array([k[0] for k in kws]), kw_variant=np.array([k[1] for k in kws]))
    print("done", lang, len(rows))


if __name__ == "__main__":
    main(sys.argv[1])
