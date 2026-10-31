import pickle
from pathlib import Path
import torch
import re
import json

from model import Poetry_Cloud, config
from tokenizer import encoder, decoder
from rhythm import rhythm_ok

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "out"
PRETRAIN = OUT_DIR / "pretrain"
SFT = OUT_DIR / "sft"

with (ROOT / "data" / "poetry_sft_v1" / "meta.pkl").open("rb") as f:
    meta = pickle.load(f)

TITLES = ["溫泉","秋夜","送別","山行","江上"]
SEEDS  = range(1000, 1020)
genre = lambda s: "<|绝句|>" if s >= 1010 else "<|律诗|>"
device = "cuda" if torch.cuda.is_available() else "cpu"
max_tokens, temperature, top_k = 80, 0.8, 40
eot_id = meta["eot_id"]
WANTS = {"<|绝句|>" : 2, "<|律诗|>" : 4}
PUNCT = "[，。？！；、]"
OUT_EVAL = OUT_DIR / "eval"
OUT_EVAL.mkdir(parents = True, exist_ok = True)

def score_one(text, genre):
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    want = WANTS.get(genre)

    fmt = (want is not None and len(lines) == want)
    len_set = set()

    for line in lines:
        parts = [p for p in re.split(PUNCT, line) if p]
        if len(parts) != 2:
            fmt = False
            break
        for p in parts:
            len_set.add(len(re.findall(r"[\u4e00-\u9fff]", p)))

    if len(lines) == (want or -1):
        fmt = fmt and len(len_set) == 1 and len_set.issubset({5, 7})

    rhythm = rhythm_ok(lines) if len(lines) in (2, 4) else None

    sents = [s for s in re.split(PUNCT, text) if s.strip()]
    dup_sents = len(sents) - len(set(sents))
    dup_adj = bool(re.search(r"([\u4e00-\u9fff])\1", text)) or \
              bool(re.search(r"([\u4e00-\u9fff]{2})\1", text)) # 检查相邻两个有没有重复

    return {"lines" : lines, "fmt_ok" : fmt, "rhythm" : rhythm,
            "dup_sents" : dup_sents, "dup_adj" : dup_adj, "n_sents" : len(sents)}

def gen_poetry(model : Poetry_Cloud, title, seed, genre):
    torch.manual_seed(seed)
    prompt = f"{genre}{title}\n"
    ids = torch.tensor([encoder(prompt, meta)], dtype = torch.long, device = device)
    out = model.generate(
        ids,
        max_tokens = max_tokens,
        temperature = temperature,
        top_k = top_k,
        stop_id = eot_id
    )
    new = out[0, ids.size(1):].tolist()
    text = decoder(new, meta).strip()
    truncated = len(new) >= max_tokens
    return text, truncated



if __name__ == "__main__":
    for model in ("sft", "pretrain"):
        ckpt = "sft" if model == "sft" else "pretrain"
        CK = torch.load((OUT_DIR / ckpt / "ckpt" / "ckptbest.pt"), map_location=device)
        poetry_cloud = Poetry_Cloud(config(**CK["model_args"])).to(device)
        poetry_cloud.load_state_dict(CK["model"])

        stats = {
            "fmt_ok" : 0,
            "rhythm_none" : 0,
            "rhythm_false" : 0,
            "rhythm_ok" : 0,
            "dup_sents" : 0,
            "dup_adj" : 0,
            "n_sent" : 0.0,
            "truncated" : 0
        }
        poetry_cloud.eval()
        n_sents_total = 0
        for title in TITLES:
            for seed in SEEDS:
                g = genre(seed)
                text, truncated = gen_poetry(poetry_cloud, title, seed, g)
                val = score_one(text, g)
                if val["fmt_ok"]:
                    stats["fmt_ok"] += 1
                if val["rhythm"] is None:
                    stats["rhythm_none"] += 1
                elif val["rhythm"] is False:
                    stats["rhythm_false"] += 1
                else:
                    stats["rhythm_ok"] += 1
                if val["dup_sents"]:
                    stats["dup_sents"] += val["dup_sents"]
                if val["dup_adj"]:
                    stats["dup_adj"] += val["dup_adj"]
                if truncated:
                    stats["truncated"] += 1
                n_sents_total += val["n_sents"]

        stats["n_sent"] = stats["dup_sents"] / n_sents_total
        (OUT_EVAL / f"{model}_stats.json").write_text(
            json.dumps(stats, ensure_ascii = False, indent = 2), encoding = "UTF-8"
        )
        print(model, CK["iter_step"], stats)