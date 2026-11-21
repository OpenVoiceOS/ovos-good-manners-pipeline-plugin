"""One-time: FLEURS parquet (one huge row group) -> one wav per utterance + texts.tsv, so the
runners don't hold a whole language in memory."""
import sys, os
from fleurs import ROOT
import pyarrow.parquet as pq, soundfile as sf
for l in sys.argv[1:]:
    d = f"{ROOT}{l}"; os.makedirs(d, exist_ok=True)
    t = pq.read_table(f"{ROOT}{l}.parquet", columns=["id", "audio", "transcription"])
    with open(f"{d}/texts.tsv", "w") as f:
        for i in range(t.num_rows):
            r = t.slice(i, 1).to_pylist()[0]
            open(f"{d}/{i:04d}.wav", "wb").write(r["audio"]["bytes"])
            f.write(f"{i:04d}\t{r['transcription']}\n")
    del t
    print(l, "ok", flush=True)
