"""Fetch what the test needs into $OVERHEARD_DATA (default tests/overheard/data, not in git):
WakePhoneHuBERT (Apache-2.0), the Piper voices in run_pos.CAST, and FLEURS test speech (CC-BY-4.0)."""
import json, os, urllib.request
from fleurs import DATA, FLEURS, ROOT
HF = "https://huggingface.co/"
os.makedirs(ROOT, exist_ok=True); os.makedirs(DATA + "/piper", exist_ok=True)
here = os.path.dirname(os.path.abspath(__file__))


def get(url, dst):
    if not os.path.exists(dst):
        print("get", dst, flush=True)
        urllib.request.urlretrieve(url, dst)


for f in ["wakephonehubert_int8.onnx", "vocab.json"]:
    get(HF + "TigreGotico/wakephonehubert/resolve/main/" + f, os.path.join(here, f))
voices = json.load(urllib.request.urlopen(HF + "rhasspy/piper-voices/resolve/main/voices.json"))
from run_pos import CAST
for model in {m for cast in CAST.values() for m, _, _ in cast}:
    for f in voices[model]["files"]:
        if f.endswith((".onnx", ".onnx.json")):
            get(HF + "rhasspy/piper-voices/resolve/main/" + f, DATA + "/piper/" + os.path.basename(f))
for code in FLEURS.values():
    get(HF + f"api/datasets/google/fleurs/parquet/{code}/test/0.parquet", ROOT + code + ".parquet")
    if not os.path.exists(ROOT + code + "/texts.tsv"):
        os.system(f"python3 {here}/extract_fleurs.py {code}")
