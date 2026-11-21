"""Go/no-go per phrase: pick the threshold that gives at most TARGET false hits per hour on normal
speech, then see how many of the spoken phrases it catches, and whether look-alikes trigger.

A phrase's score is the max over its variants (English phrases: English and local pronunciation).
False hits closer than 2 s count once, like the model card."""
import json, sys
from collections import defaultdict
import numpy as np

TARGETS = [1.0, 0.1]          # false hits per hour per phrase
BLOCK_S = 0.08


def phrase_matrix(scores, kw_phrase):
    names = list(dict.fromkeys(kw_phrase.tolist()))
    cols = [np.max(scores[:, kw_phrase == n], axis=1) for n in names]
    return names, np.stack(cols, 1)


def events(series, utt, thr, gap=int(2 / BLOCK_S)):
    """Number of separate hits (>= thr), merging hits < 2 s apart in the same utterance."""
    idx = np.flatnonzero(series >= thr)
    if not len(idx):
        return 0, []
    n, starts = 1, [idx[0]]
    for a, b in zip(idx[:-1], idx[1:]):
        if b - a > gap or utt[a] != utt[b]:
            n += 1; starts.append(b)
    return n, starts


def threshold_for(series, utt, hours, target):
    """Lowest threshold whose false-hit rate is <= target per hour."""
    allowed = int(np.floor(target * hours))
    cand = np.unique(np.round(series[series > -3], 3))[::-1]
    if not len(cand):
        return -3.0
    # coarse-to-fine search from strict to lenient
    best = cand[0] + 1e-3
    lo, hi = 0, len(cand) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        n, _ = events(series, utt, cand[mid])
        if n <= allowed:
            best = cand[mid]; lo = mid + 1
        else:
            hi = mid - 1
    return float(best)


def main(langs):
    out = {}
    for lang in langs:
        neg = np.load(f"scores/neg_{lang}.npz")
        pos = np.load(f"scores/pos_{lang}.npz")
        hours = len(neg["scores"]) * BLOCK_S / 3600
        names, N = phrase_matrix(neg["scores"], neg["kw_phrase"])
        _, Pk = phrase_matrix(pos["peak"], pos["kw_phrase"])
        meta = [json.loads(m) for m in pos["meta"]]
        res = {"hours": round(hours, 2), "phrases": {}}
        for j, name in enumerate(names):
            r = {}
            for target in TARGETS:
                thr = threshold_for(N[:, j], neg["utt"], hours, target)
                hits = [m for m, p in zip(meta, Pk[:, j]) if m["kind"] == "pos" and m["target"] == name]
                caught = [p >= thr for m, p in zip(meta, Pk[:, j]) if m["kind"] == "pos" and m["target"] == name]
                by_cond = defaultdict(list)
                for m, c in zip(hits, caught):
                    by_cond[m["cond"]].append(c)
                # look-alikes and the other phrases that trigger this one
                look = [m["text"] for m, p in zip(meta, Pk[:, j]) if m["kind"] == "look" and p >= thr]
                other = [m["text"] for m, p in zip(meta, Pk[:, j])
                         if m["kind"] == "pos" and m["target"] != name and p >= thr]
                n_look = sum(1 for m in meta if m["kind"] == "look")
                r[str(target)] = dict(
                    threshold=round(thr, 3),
                    recall=round(float(np.mean(caught)), 3) if caught else None, n=len(caught),
                    recall_by_cond={k: round(float(np.mean(v)), 3) for k, v in by_cond.items()},
                    lookalike_rate=round(len(look) / max(1, n_look), 3), lookalikes_hit=sorted(set(look)),
                    other_phrases_hit=sorted(set(other)))
            # what normal speech made it fire at 1/h
            thr = r["1.0"]["threshold"]
            _, starts = events(N[:, j], neg["utt"], thr)
            r["fired_on"] = [str(neg["texts"][neg["utt"][s]])[:90] for s in starts[:3]]
            res["phrases"][name] = r
        out[lang] = res
    json.dump(out, open("scores/report.json", "w"), ensure_ascii=False, indent=1)
    for lang, res in out.items():
        print(f"\n== {lang}  ({res['hours']} h normal speech)")
        print(f"{'phrase':24s} {'recall@1/h':>10s} {'@0.1/h':>7s}  clean/room/noise/babble @1/h   lookalikes@1/h")
        for name, r in res["phrases"].items():
            a, b = r["1.0"], r["0.1"]
            c = a["recall_by_cond"]
            conds = "/".join(f"{c.get(k, float('nan')):.2f}" for k in ["clean", "room", "noise10", "babble10"])
            print(f"{name:24s} {a['recall']:>10.2f} {b['recall']:>7.2f}  {conds:28s} {a['lookalike_rate']:.2f} {a['lookalikes_hit'][:2]}")


if __name__ == "__main__":
    main(sys.argv[1:])
