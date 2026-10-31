import json
import pickle
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

from rhythm import rhythm_ok

ROOT = Path(__file__).resolve().parent

DATA_SET = ROOT / "data" / "全唐诗"
OUT_DIR = ROOT / "data" / "poetry_char_v3"
GENRE = {2 : "<|绝句|>", 4 : "<|律诗|>"}

def iter_poems():
    for pattens in ("poet.song.*.json", "poet.tang.*.json"):
        for path in sorted(DATA_SET.glob(pattens)):
            with path.open("r", encoding = "UTF-8") as f:
                for poem in json.load(f):
                    yield poem

def clean_line(s):
    return re.sub(r"[^\u4e00-\u9fff]", "", s)

def is_quality(paragraph):
    if len(paragraph) != 2 and len(paragraph) != 4:
        return False
    length = [clean_line(p) for p in paragraph]
    lens = {len(x) for x in length}
    ok = len(lens) == 1 and lens.pop() in (10, 14)
    return ["<|绝句|>" if len(paragraph) == 2 else "<|律诗|>", ok]

def collect_poems(limit = None):
    seen = set()
    records = list()
    stats = {
        "raw" : 0,
        "four_poems" : 0,
        "eight_poems" : 0,
        "strict" : 0,
        "duplicate" : 0,
        "rhythm_unknown" : 0,
        "rhythm_fail" : 0,
    }

    for i, poem in enumerate(iter_poems(), start = 1):
        if limit is not None and i > limit:
            break
        stats["raw"] += 1
        paragraph = poem.get("paragraphs") or []

        if len(paragraph) != 2 and len(paragraph) != 4:
            continue
        genre, ok = is_quality(paragraph)
        if genre == "<|绝句|>":
            stats["four_poems"] += 1
        else:
            stats["eight_poems"] += 1

        verdict = rhythm_ok(paragraph)
        if verdict == None:
            stats["rhythm_unknown"] += 1
            continue
        if verdict == False:
            stats["rhythm_fail"] += 1
            continue

        if not ok:
            continue
        stats["strict"] += 1

        key = "".join(clean_line(p) for p in paragraph)
        if key in seen:
            stats["duplicate"] += 1
            continue
        seen.add(key)

        records.append(
            {"genre" : genre,
             "title" : clean_line(poem.get("title", "")),
             "paragraph" : list(paragraph)}
        )

    return stats, records

def poem_to_text(record):
    return (record["genre"] +
        record["title"] + "\n"
        + "\n".join(record["paragraph"]) + "\n"
    )

def encoder(text, stoi):
    return [stoi.get(ch, unk_id) for ch in text]

def join_tokens(indices):
    idx = []
    for i in indices:
        idx.extend(encoder(text[i], stoi))
        idx.append(eot_id)
    return np.asarray(idx, dtype = np.uint16)

def first_key(rec):
    p = clean_line(rec["paragraph"][0])
    width = 5 if len(p) == 10 else 7
    return (width, p[:width])

if __name__ == "__main__":
    MAX_TOKEN = 128

    stats, record = collect_poems()
    print(stats)
    print("unique record:", len(record))

    before_record = len(record)

    record = [
        p for p in record if len(poem_to_text(p)) + 1 <= MAX_TOKEN
    ]

    print("drop long:", before_record - len(record))
    print("now long:", len(record))

    text = [poem_to_text(p) for p in record]
    all_char = "".join(text)
    char_count = Counter(all_char)

    min_freq = 2
    itos = ["<unk>", "<|endoftext|>"] + sorted(ch for ch, n in char_count.items() if n >= min_freq)
    stoi = {ch : i for i, ch in enumerate(itos)}

    assert not any(c.isascii() and c.isalpha() for c in itos[2:])
    print(len(itos))

    groups = defaultdict(list)
    for i, x in enumerate(record):
        groups[first_key(x)].append(i)

    keys = sorted(groups)
    random.Random(42).shuffle(keys)
    cut = int(len(keys) * 0.99)
    train_idx = [i for k in keys[:cut] for i in groups[k]]
    val_idx =[i for k in keys[cut:] for i in groups[k]]

    assert len(train_idx) + len(val_idx) == len(record)
    assert not (set(keys[cut:]) & set(keys[:cut]))

    multi = {k: v for k, v in groups.items() if len(v) > 1}
    print("groups", len(groups), "multi", len(multi), "poems", sum(len(v) for v in multi.values()))
    print("top3", sorted(multi.items(), key=lambda kv: -len(kv[1]))[:3])
    print("train", len(train_idx), "val", len(val_idx))

    unk_id = stoi["<unk>"]
    eot_id = stoi["<|endoftext|>"]

    train_ids = join_tokens(train_idx)
    val_ids = join_tokens(val_idx)

    OUT_DIR.mkdir(parents = True, exist_ok = True)

    train_ids.tofile(OUT_DIR / "train.bin")
    val_ids.tofile(OUT_DIR / "val.bin")

    meta = {
        "vocab_size" : len(itos),
        "stoi" : stoi,
        "itos" : itos,
        "min_freq" : min_freq,
        "seed" : 42,
        "format" : "<genre><title>\n<body>\n<|endoftext|>",
        "unk_id" : unk_id,
        "eot_id" : eot_id
    }

    with (OUT_DIR / "meta.pkl").open("wb") as f:
        pickle.dump(meta, f)