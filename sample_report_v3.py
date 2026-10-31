import pickle
from pathlib import Path
import numpy as np
import json
import random

from rhythm import rhythm_ok

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "poetry_char_v3"
OUT_DIR = ROOT / "out" / "pretrain"
ids_path = OUT_DIR / "c3_sample_ids.json"
report_path = OUT_DIR / "c3_sample_report.json"
manual_path = OUT_DIR / "c3_manual_20.txt"

with (DATA_DIR / "meta.pkl").open("rb") as f:
    meta = pickle.load(f)

val = np.memmap(
    DATA_DIR / "val.bin",
    dtype = np.uint16,
    mode = "r"
)
eot_id = int(meta["eot_id"])

cuts = np.flatnonzero(val == eot_id)
starts = np.r_[0, cuts[:-1] + 1]
ends = cuts

if ids_path.exists():
    sample_ids = json.loads(ids_path.read_text(encoding = "UTF-8"))
else:
    sample_ids = sorted(
        random.Random(42).sample(range(len(cuts)), 500)
    )
    ids_path.write_text(
        json.dumps(sample_ids, ensure_ascii = False),
        encoding = "UTF-8"
    )

stats = {
    "total" : len(sample_ids),
    "True" : 0,
    "None" : 0,
    "False" : 0,
    "None_rate" : 0.0,
    "False_rate" : 0.0,
    "seed" : 42,
    "source" : str(DATA_DIR / "val.bin"),
    "sample_path" : str(ids_path)
}

false_case = []
none_case = []

for i in sample_ids:
    ids = val[starts[i] : ends[i]].tolist()
    text = "".join(meta["itos"][x] for x in ids)

    lines = text.strip().split('\n')
    body = lines[1:]
    verdict = rhythm_ok(body)

    if verdict is True:
        stats["True"] += 1
    elif verdict is None:
        stats["None"] += 1
        if len(none_case) < 10:
            none_case.append((i, text.strip()))
    else:
        stats["False"] += 1
        if len(false_case) < 10:
            false_case.append((i, text.strip()))

stats["None_rate"] = stats["None"] / stats["total"]
stats["False_rate"] = stats["False"] / stats["total"]
print(stats)

report_path.write_text(
    json.dumps(stats, ensure_ascii = True),
    encoding = "UTF-8"
)

with manual_path.open("w", encoding = "UTF-8") as f:
    for label, case in (("False", false_case), ("None", none_case)):
        f.write(f"===={label}====\n")
        for n, (poem_id, text) in enumerate(case, start = 1):
            f.write(f"---{label}{n} | poem_id : {poem_id}---\n")
            f.write(f"{text}\n\n")