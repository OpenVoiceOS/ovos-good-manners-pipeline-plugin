"""Experiment: keywords taken from what the model actually hears, instead of eSpeak's phones.

eSpeak's Danish phones are far from what WakePhoneHuBERT hears ("av for helvede": eSpeak
ɑ u f ʌ h ɛ l ʋ ə ð ə, the model hears a l f o h i v ə). So each phrase is spoken by a few
ENROLMENT voices, the model's greedy phones become extra keyword variants, and recall is only
measured on the OTHER voices (no leakage). Negatives are re-scored with the new variants.
-> scores/enroll_<lang>.npz"""
import io, json, sys
import numpy as np, pyarrow.parquet as pq, soundfile as sf
from piper import PiperVoice, SynthesisConfig
import wph, run_pos
from fleurs import utterances
from keywords import keywords, P

ENROL = {"da": [("da_DK-talesyntese-medium", None), ("no_NO-nvcc-medium", 0), ("no_NO-nvcc-medium", 5)],
         "en": [("en_US-lessac-medium", None), ("en_US-libritts_r-medium", 100), ("en_GB-vctk-medium", 7)]}


def heard(sp, x, penalty=2.0):
    ipa = sp.sess.run(["ipa"], {"waveform": x[None]})[0][0]
    p = np.log(ipa + 1e-10); p[:, 0] -= penalty
    out, prev = [], -1
    for k in p.argmax(1):
        if k > 3 and k != prev:
            out.append(int(k))
        prev = k
    return out


def main(lang):
    rng = np.random.default_rng(7)
    sp = wph.Spotter()
    sil = np.zeros(9600, np.float32)
    phrases = P[lang]["phrases"] + ([] if lang == "en" else P["en"]["phrases"])
    enrol_ids = set()
    variants = {ph: [] for ph in phrases}
    for model, spk in ENROL[lang]:
        v = PiperVoice.load(run_pos.VOICES + model + ".onnx")
        if not model.startswith(lang[:2]) and lang != "en":
            v.config.espeak_voice = P[lang]["espeak"]
        enrol_ids.add(model if spk is None else f"{model}#{spk}")
        for ph in phrases:
            for ls in (0.9, 1.1):
                a = np.concatenate([c.audio_float_array for c in v.synthesize(ph, SynthesisConfig(speaker_id=spk, length_scale=ls))])
                h = heard(sp, np.concatenate([sil, run_pos.to16k(a, v.config.sample_rate), sil]))
                if len(h) >= 3 and h not in variants[ph]:
                    variants[ph].append(h)
    base = keywords(lang)
    kws = base + [(ph, "heard", h) for ph, hs in variants.items() for h in hs]
    for ph, hs in variants.items():
        print(ph, "|", " / ".join(" ".join(wph.SYMBOLS[i] for i in h) for h in hs), flush=True)

    # positives: the run_pos cast minus the enrolment voices, regenerated with the same seed logic
    rows = []
    babble = [a for a, _ in utterances(lang, last=30)]
    texts = [(ph, "pos", ph) for ph in phrases] + [(s, "look", "") for s in P[lang]["lookalikes"]]
    for model, nspk, override in run_pos.CAST[lang]:
        v = PiperVoice.load(run_pos.VOICES + model + ".onnx")
        if override:
            v.config.espeak_voice = override
        spks = [None] if not nspk else rng.choice(v.config.num_speakers, nspk + 2, replace=False).tolist()
        for spk in spks:
            vid = model if spk is None else f"{model}#{spk}"
            if vid in enrol_ids:
                continue
            for text, kind, target in texts:
                a = np.concatenate([c.audio_float_array for c in v.synthesize(text, SynthesisConfig(speaker_id=spk, length_scale=float(rng.uniform(0.85, 1.25))))])
                clip = np.concatenate([sil, run_pos.to16k(a, v.config.sample_rate), sil])
                for cond in run_pos.CONDITIONS:
                    x = run_pos.condition(clip, cond, rng, babble)
                    x = (x / (np.abs(x).max() + 1e-9) * rng.uniform(0.2, 0.8)).astype(np.float32)
                    s, _ = sp.stream(x, [k[2] for k in kws])
                    rows.append((dict(voice=vid, text=text, kind=kind, target=target, cond=cond), s.max(0).astype(np.float32)))
        print("pos", model, len(rows), flush=True)

    # negatives
    S, U, texts_n = [], [], []
    pad = np.zeros(8000, np.float32)
    for a, text in utterances(lang):
        if True:
            s, _ = sp.stream(np.concatenate([pad, a, pad]), [k[2] for k in kws])
            S.append(s.astype(np.float32)); U.append(np.full(len(s), len(texts_n))); texts_n.append(text)
        if len(texts_n) % 160 == 0:
            print("neg", len(texts_n), flush=True)
    np.savez_compressed(f"scores/enroll_{lang}.npz",
                        pos_peak=np.stack([r[1] for r in rows]), pos_meta=np.array([json.dumps(r[0], ensure_ascii=False) for r in rows]),
                        scores=np.concatenate(S), utt=np.concatenate(U), texts=np.array(texts_n),
                        kw_phrase=np.array([k[0] for k in kws]), kw_variant=np.array([k[1] for k in kws]),
                        kw_phones=np.array([" ".join(wph.SYMBOLS[i] for i in k[2]) for k in kws]))
    print("done", lang)


if __name__ == "__main__":
    main(sys.argv[1])
