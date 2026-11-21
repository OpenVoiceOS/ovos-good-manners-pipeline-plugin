"""Two stages: WakePhoneHuBERT set sensitive (STAGE1 false hits per hour per phrase), and when it
fires, local STT (faster-whisper) on the last few seconds; good-manners' own word lists decide on
the text whether anything foul was said. Only then would Emkay comment.

Measures, per language:
  - false comments per hour on normal speech (FLEURS), and how often STT even runs
  - recall on spoken phrases (voices not used anywhere else in this script's choices), by condition
-> scores/stage2_<lang>.json"""
import json, re, sys, time
from pathlib import Path
import numpy as np
from faster_whisper import WhisperModel
from piper import PiperVoice, SynthesisConfig
import wph, run_pos
from keywords import keywords, P
from fleurs import utterances
from report import phrase_matrix, threshold_for, events, BLOCK_S

STAGE1 = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
LOCALE = Path(__file__).resolve().parents[2] / "ovos_good_manners_pipeline_plugin" / "locale"
GM = {"da": "da-dk", "en": "en-us", "de": "de-de"}
WHISPER_LANG = {"da": "da", "en": "en", "de": "de"}
BEFORE, AFTER = 2.5, 0.5          # seconds of audio given to STT around a hit


def voc(lang, name):
    f = LOCALE / GM[lang] / f"{name}.voc"
    return [l.strip().lower() for l in f.read_text().splitlines() if l.strip() and not l.startswith("#")] if f.exists() else []


def matcher(lang):
    """Same matching as the plugin: whole words/phrases, plus foul prefixes (lortebil). The English
    list is always included, as people swear in English everywhere."""
    words = set(voc(lang, "foul_language") + voc(lang, "slurs") + voc("en", "foul_language") + voc("en", "slurs"))
    words |= {p.lower() for p in P[lang]["phrases"] + P["en"]["phrases"]}
    pre = set(voc(lang, "foul_prefixes") + voc("en", "foul_prefixes"))
    w = re.compile(r"(?<!\w)(?:" + "|".join(map(re.escape, sorted(words, key=len, reverse=True))) + r")(?!\w)")
    p = re.compile(r"(?<!\w)(?:" + "|".join(map(re.escape, sorted(pre, key=len, reverse=True))) + r")\w*") if pre else None
    censored = re.compile(r"\w\*+\w*")           # Whisper writes "f**k"
    return lambda text: bool(w.search(text.lower()) or (p and p.search(text.lower())) or censored.search(text))


def main(lang):
    t0 = time.time()
    stt = WhisperModel("small", device="cpu", compute_type="int8", cpu_threads=2)
    is_foul = matcher(lang)

    def confirm(x):
        segs, _ = stt.transcribe(x, language=WHISPER_LANG[lang], beam_size=1, vad_filter=False,
                                 condition_on_previous_text=False)
        text = " ".join(s.text for s in segs).strip()
        return is_foul(text), text

    # stage-1 thresholds from the negatives already scored
    neg = np.load(f"scores/neg_{lang}.npz")
    names, N = phrase_matrix(neg["scores"], neg["kw_phrase"])
    hours = len(N) * BLOCK_S / 3600
    thr = np.array([threshold_for(N[:, j], neg["utt"], hours, STAGE1) for j in range(len(names))])
    fired = (N >= thr).any(1).astype(float)
    n_ev, starts = events(fired, neg["utt"], 0.5)

    # false comments on normal speech: run STT on every stage-1 hit
    audio = [a for a, _ in utterances(lang)]
    first_block = np.r_[0, np.flatnonzero(np.diff(neg["utt"])) + 1]
    false_comments = []
    for s in starts:
        u = neg["utt"][s]
        end = (s - first_block[u] + 1) * 1280 - 8000          # the runner padded 0.5 s in front
        x = audio[u][max(0, int(end - BEFORE * 16000)):int(end + AFTER * 16000)]
        foul, text = confirm(x.astype(np.float32))
        if foul:
            false_comments.append(dict(heard=text, said=str(neg["texts"][u])[:120]))
    print(lang, f"stage1 {n_ev} hits in {hours:.2f} h, false comments {len(false_comments)}", f"{time.time()-t0:.0f}s", flush=True)

    # positives: fresh clips from a few voices, through both stages
    rng = np.random.default_rng(99)
    sp = wph.Spotter()
    kws = keywords(lang)
    kw_phrase = np.array([k[0] for k in kws])
    babble = [a for a, _ in utterances(lang, last=30)]
    sil = np.zeros(9600, np.float32)
    cast = run_pos.CAST[lang]
    rows = []
    for model, nspk, override in cast[:4]:
        v = PiperVoice.load(run_pos.VOICES + model + ".onnx")
        if override:
            v.config.espeak_voice = override
        spk = None if not nspk else int(rng.integers(v.config.num_speakers))
        for ph in P[lang]["phrases"] + ([] if lang == "en" else P["en"]["phrases"]):
            a = np.concatenate([c.audio_float_array for c in v.synthesize(ph, SynthesisConfig(speaker_id=spk, length_scale=float(rng.uniform(0.85, 1.25))))])
            clip = np.concatenate([sil, run_pos.to16k(a, v.config.sample_rate), sil])
            for cond in run_pos.CONDITIONS:
                x = run_pos.condition(clip, cond, rng, babble)
                x = (x / (np.abs(x).max() + 1e-9) * rng.uniform(0.2, 0.8)).astype(np.float32)
                s, _ = sp.stream(x, [k[2] for k in kws])
                _, M = phrase_matrix(s, kw_phrase)
                hit1 = bool((M >= thr).any())        # same phrase order as the negatives
                foul, text = confirm(x) if hit1 else (False, "")
                rows.append(dict(voice=model, phrase=ph, cond=cond, stage1=hit1, comment=hit1 and foul, heard=text))
        print(lang, model, len(rows), f"{time.time()-t0:.0f}s", flush=True)
        del v

    rec = lambda rs: round(float(np.mean([r["comment"] for r in rs])), 3) if rs else None
    out = dict(lang=lang, stage1_per_phrase_per_hour=STAGE1, hours=round(hours, 2), stage1_hits=n_ev,
               stage1_hits_per_hour=round(n_ev / hours, 1), false_comments=len(false_comments),
               false_comments_per_hour=round(len(false_comments) / hours, 3), false_comment_examples=false_comments[:10],
               recall=rec(rows), recall_stage1=round(float(np.mean([r["stage1"] for r in rows])), 3),
               recall_by_cond={c: rec([r for r in rows if r["cond"] == c]) for c in run_pos.CONDITIONS},
               recall_by_phrase={ph: rec([r for r in rows if r["phrase"] == ph]) for ph in dict.fromkeys(r["phrase"] for r in rows)},
               missed_examples=[(r["phrase"], r["cond"], r["heard"]) for r in rows if r["stage1"] and not r["comment"]][:15])
    json.dump(out, open(f"scores/stage2_{lang}.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps({k: out[k] for k in ["stage1_hits_per_hour", "false_comments_per_hour", "recall", "recall_stage1", "recall_by_cond"]}))


if __name__ == "__main__":
    main(sys.argv[1])
