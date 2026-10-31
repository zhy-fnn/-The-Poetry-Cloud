import pickle
from pathlib import Path
from collections import defaultdict
import random
import numpy as np
import json

from load import collect_poems, poem_to_text, first_key
from tokenizer import encoder

def join_tokens(records, meta):
    idx = []
    for rec in records:
        idx.extend(encoder(poem_to_text(rec), meta))
        idx.append(eot_id)
    return np.asarray(idx, dtype = np.uint16)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
SFT_DIR = DATA_DIR / "poetry_sft_v1"
SFT_DIR.mkdir(parents = True, exist_ok = True)

stats, record = collect_poems()
MAX_TOKEN = 128
record = [
    p for p in record if len(poem_to_text(p)) + 1 <= MAX_TOKEN
]
rng = random.Random(42)
five_four = rng.sample([p for p in record if first_key(p)[0] == 5 and len(p["paragraph"]) == 2], 5000)
five_eight = rng.sample([p for p in record if first_key(p)[0] == 5 and len(p["paragraph"]) == 4], 5000)
seven_four = rng.sample([p for p in record if first_key(p)[0] == 7 and len(p["paragraph"]) == 2], 5000)
seven_eight = rng.sample([p for p in record if first_key(p)[0] == 7 and len(p["paragraph"]) == 4], 5000)
chosen = five_four + five_eight + seven_four + seven_eight

with (DATA_DIR / "poetry_char_v3" / "meta.pkl").open("rb") as f:
    meta = pickle.load(f)

assert len(chosen) == 20000

text = (poem_to_text(p) for p in record)
itos = meta["itos"]
stoi = meta["stoi"]
eot_id = meta["eot_id"]

groups = defaultdict(list)
for i, x in enumerate(chosen):
    groups[first_key(x)].append(i)

keys = sorted(groups)
rng.shuffle(keys)
cuts = int(len(keys) * 0.99)
train_data = [i for k in keys[:cuts] for i in groups[k]]
val_data = [i for k in keys[cuts:] for i in groups[k]]

train_records = [chosen[i] for i in train_data]
val_records = [chosen[i] for i in val_data]

train_keys = {first_key(rec) for rec in train_records}
val_keys = {first_key(rec) for rec in val_records}

assert train_keys.isdisjoint(val_keys)

print("test passed")


train_ids = join_tokens(train_records, meta)
val_ids = join_tokens(val_records, meta)

train_ids.tofile(SFT_DIR / "train.bin")
val_ids.tofile(SFT_DIR / "val.bin")

sft_stats = {
    "seed" : 42,
    "total" : len(chosen),
    "counts" : {
    "five_jueju" : len(five_four),
    "five_lushi" : len(five_eight),
    "seven_juelu" : len(seven_four),
    "seven_lushi" : len(seven_eight)
    },
    "train_data" : len(train_records),
    "val_data" : len(val_records),
    "train_tokens" : len(train_ids),
    "val_tokens" : len(val_ids),
    "source" : str(DATA_DIR / "poetry_char_v3")
}

print(sft_stats)

with (SFT_DIR / "meta.pkl").open("wb") as f:
    pickle.dump(meta, f)

with (SFT_DIR / "sft_manifest.json").open("w", encoding = "UTF-8") as f:
    json.dump(sft_stats, f)