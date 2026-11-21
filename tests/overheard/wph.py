"""WakePhoneHuBERT zero-shot phrase spotting, as the model card and wph.js do it:
1.5 s window, scored after every 80 ms block, best CTC path / n phones."""
import json
from pathlib import Path
import numpy as np
import onnxruntime as ort

HERE = Path(__file__).parent
WINDOW, BLOCK = 24000, 1280
SYMBOLS = json.load(open(HERE / "vocab.json"))["symbols"]
_INDEX = {p: i for i, p in enumerate(SYMBOLS) if i > 3}
_MAXLEN = max(len(p) for p in SYMBOLS[4:])

_espeak = {}
# eSpeak symbols that vocab.json spells differently (Greek epsilon in Danish)
_FIX = str.maketrans({"ε": "ɛ"})
def phones(text, lang):
    import espeakng_loader
    from phonemizer.backend import EspeakBackend
    from phonemizer.backend.espeak.wrapper import EspeakWrapper
    from phonemizer.separator import Separator
    if lang not in _espeak:
        EspeakWrapper.set_library(espeakng_loader.get_library_path())
        EspeakWrapper.set_data_path(espeakng_loader.get_data_path())
        _espeak[lang] = EspeakBackend(lang, language_switch="remove-flags")
    raw = _espeak[lang].phonemize([text], separator=Separator(phone=" ", word=" | ", syllable=""), strip=True)[0].replace("|", " ").split()
    ids = []
    for p in raw:
        p = p.translate(_FIX)
        if p in _INDEX:
            ids.append(_INDEX[p]); continue
        i = 0                                   # longest match, like g2p.js
        while i < len(p):
            n = min(_MAXLEN, len(p) - i)
            while n and p[i:i + n] not in _INDEX: n -= 1
            if not n: raise ValueError(f"{text!r}: phone {p!r} not in vocab")
            ids.append(_INDEX[p[i:i + n]]); i += n
    return ids

def compile_kws(kws):
    """Pad all keywords' CTC label sequences into one batch, so every window is scored for all of
    them in one pass over its frames."""
    labs = [np.array([0] + [x for k in kw for x in (k, 0)]) for kw in kws]
    S = max(len(l) for l in labs); K = len(kws)
    lab = np.zeros((K, S), int); valid = np.zeros((K, S), bool); skip = np.zeros((K, S), bool)
    last = np.array([len(l) - 1 for l in labs]); n = np.array([len(kw) for kw in kws], float)
    for k, l in enumerate(labs):
        lab[k, :len(l)] = l; valid[k, :len(l)] = True
        for s in range(2, len(l)):
            skip[k, s] = l[s] != 0 and l[s] != l[s - 2]
    return lab, valid, skip, last, n

def score_many(logr, ck):
    """Best-path score of every keyword in one window; same numbers as the model card's keyword_score."""
    lab, valid, skip, last, n = ck
    K, S = lab.shape
    e = np.where(valid[None], logr[:, lab], -np.inf)        # [T,K,S]
    a = np.full((K, S), -np.inf); best = np.full(K, -np.inf)
    ninf1 = np.full((K, 1), -np.inf); ninf2 = np.full((K, 2), -np.inf)
    rows = np.arange(K)
    for t in range(e.shape[0]):
        p = np.maximum(a, np.hstack([ninf1, a[:, :-1]]))
        p = np.maximum(p, np.where(skip, np.hstack([ninf2, a[:, :-2]]), -np.inf))
        p[:, :2] = np.maximum(p[:, :2], 0.0)
        a = p + e[t]
        best = np.maximum(best, np.maximum(a[rows, last], a[rows, last - 1]))
    return best / n

class Spotter:
    def __init__(self, threads=1):
        o = ort.SessionOptions(); o.intra_op_num_threads = threads; o.inter_op_num_threads = 1; o.enable_cpu_mem_arena = False
        self.sess = ort.InferenceSession(str(HERE / "wakephonehubert_int8.onnx"), o, providers=["CPUExecutionProvider"])

    def stream(self, audio, kws):
        """Scores [n_blocks, n_kw] for audio streamed in 80 ms blocks, plus vad per block (max of new frames)."""
        buf = np.zeros(WINDOW, np.float32)
        n = len(audio) // BLOCK
        ck = compile_kws(kws)
        scores = np.empty((n, len(kws))); vads = np.empty(n)
        for i in range(n):
            buf = np.concatenate([buf[BLOCK:], audio[i * BLOCK:(i + 1) * BLOCK]])
            vad, ipa = self.sess.run(["vad", "ipa"], {"waveform": buf[None]})
            r = np.log(ipa[0] + 1e-10); r -= r.max(1, keepdims=True)
            scores[i] = score_many(r, ck)
            vads[i] = vad[0, -4:, 0].max()
        return scores, vads
