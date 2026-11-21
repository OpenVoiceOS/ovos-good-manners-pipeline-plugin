"""Score every keyword over FLEURS test speech of one language (no swearing in it) -> scores/neg_<lang>.npz"""
import sys, time
import numpy as np
import wph
from fleurs import utterances
from keywords import keywords

lang = sys.argv[1]
kws = keywords(lang)
sp = wph.Spotter()
seen = set(); S = []; V = []; U = []; texts = []
pad = np.zeros(8000, np.float32)
t0 = time.time()
if True:
    for a, text in utterances(lang):
        s, v = sp.stream(np.concatenate([pad, a, pad]), [k[2] for k in kws])
        S.append(s.astype(np.float32)); V.append(v.astype(np.float32)); U.append(np.full(len(s), len(texts)))
        texts.append(text)
    if len(texts) % 160 == 0: print(lang, len(texts), f"{time.time()-t0:.0f}s", flush=True)
np.savez_compressed(f"scores/neg_{lang}.npz", scores=np.concatenate(S), vad=np.concatenate(V),
                    utt=np.concatenate(U), texts=np.array(texts), kw_phrase=np.array([k[0] for k in kws]),
                    kw_variant=np.array([k[1] for k in kws]))
print("done", lang, f"{time.time()-t0:.0f}s")
