import pickle
from pathlib import Path
import numpy as np
import re
import torch

from tokenizer import encoder, decoder

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "poetry_char_v3"

with (DATA_DIR / "meta.pkl").open("rb") as f:
    meta = pickle.load(f)

train_data = np.memmap(
    DATA_DIR / "train.bin",
    dtype = np.uint16,
    mode = "r"
)

val_data = np.memmap(
    DATA_DIR / "val.bin",
    dtype = np.uint16,
    mode = "r"
)
eot_id = int(meta["eot_id"])

train_end = np.flatnonzero(train_data == eot_id)
train_start = np.r_[0, train_end[:-1] + 1]
val_end = np.flatnonzero(val_data == eot_id)
val_start = np.r_[0, val_end[:-1] + 1]

for i in range(100):
    ids = train_data[train_start[i]: train_end[i]].astype(int).tolist()
    text = decoder(ids, meta)
    assert ids == encoder(text, meta)

print("test 1 passed")

assert int((train_data == eot_id).sum()) == 223947
assert int((val_data == eot_id).sum()) == 2278
print("test 2 passed")

def first_key(data, start, end):
    text = decoder(
        data[start:end].astype(int).tolist(),
        meta
    )
    first_body = text.strip().split("\n")[1]
    clean = re.sub(r"[^\u4e00-\u9fff]", "", first_body)
    width = 5 if len(clean) == 10 else 7
    return width, clean[:width]

train_keys = set()
val_keys = set()
for data, keys, starts, ends in (
        (train_data, train_keys, train_start, train_end),
        (val_data, val_keys, val_start, val_end)):
    for s, e in zip(starts, ends):
        keys.add(first_key(data, int(s), int(e)))

assert train_keys.isdisjoint(val_keys)
print("test 3 passed")

ix = torch.randint(
    len(train_data) - 128,
    (32,)
).tolist()

x = torch.stack([
    torch.from_numpy(train_data[i : i + 128].astype(np.int64))
    for i in ix
])

y = torch.stack([
    torch.from_numpy(train_data[i + 1 : i + 1 + 128].astype(np.int64))
    for i in ix
])

assert x.shape == (32, 128)
assert y.shape == (32, 128)
assert x.dtype == torch.int64
assert y.dtype == torch.int64
print("test 4 passed")

assert torch.equal(x[:, 1:],y[:, :-1])
print("test 5 passed")
print("test 5/5")