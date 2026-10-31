import pickle
from pathlib import Path

def encoder(text, meta):
    return[
        meta["stoi"].get(ch, meta["unk_id"])
        for ch in text
    ]

def decoder(idx, meta):
    return "".join(
        meta["itos"][i]
        for i in idx
    )

ROOT = Path(__file__).resolve().parent

stage = "pretrain"
OUT_DIR = ROOT / "out" / stage
DATA_DIR = ROOT / "data" /"poetry_char_v2"
with (DATA_DIR / "meta.pkl").open("rb") as f:
    meta = pickle.load(f)