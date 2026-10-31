import json
import sys
from pathlib import Path
import torch

from sample import load_meta, load_model, make_poem
from eval_compare import score_one

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out" / "best_of_n"

CKPT = ROOT / "out" / "pretrain" / "ckpt" / "ckptbest.pt"
TITLES = ["溫泉", "秋夜", "送別", "山行", "江上"]
GENRE = "<|绝句|>"
SEED_BASE = 2000

def best_of_n(
        ckpt = CKPT,
        n_per_titles = 30,
        smoke = False,
        out_dir = OUT
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    titles = TITLES[:1] if smoke else TITLES
    n_per_titles = n_per_titles if smoke is False else 1
    out_path = out_dir / ("smoke.jsonl" if smoke else "candidates.jsonl")

    meta = load_meta()
    model, ck = load_model(ckpt)
    rows = []
    with out_path.open("w", encoding = "UTF-8") as f:
        for ti, title in enumerate(titles):
            for j in range(n_per_titles):
                seed = SEED_BASE + ti * 100 + j
                with torch.no_grad():
                    body = make_poem(
                        model, meta, title, GENRE, seed = seed
                    ).strip()

                score = score_one(body, GENRE)
                row = {
                    "title" : title,
                    "genre" : GENRE,
                    "seed" : seed,
                    "body" : body,
                    **score
                }
                f.write(json.dumps(row, ensure_ascii = False) + "\n")
                f.flush()
                rows.append(row)
                print(title, seed, score["fmt_ok"], score["rhythm"])

    if smoke:
        return None

    summary = {"ckpt" : str(ckpt), "title" : {}, "missing" : []}
    select = []
    for title in TITLES:
        group = [r for r in rows if r["title"] == title]
        eligible = [
           r for r in group if r["fmt_ok"] and r["rhythm"] is True
        ]

        pick = min(
            eligible,
           key = lambda r : (r["dup_adj"], r["seed"])
        )if eligible else None

        summary["title"][title] = {
            "n" : len(group),
            "fmt_ok" : sum(r["fmt_ok"] for r in group),
            "rhythm_true" : sum(r["rhythm"] is True for r in group),
            "rhythm_false" : sum(r["rhythm"] is False for r in group),
            "rhythm_none" : sum(r["rhythm"] is None for r in group),
            "select_seed" : pick["seed"] if pick else None,
        }

        if pick:
            select.append(pick)
        else:
            summary["missing"].append(title)

    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii = False, indent = 2),
        encoding = "UTF-8"
    )

    gallery = ["# Poetry Cloud · Best-of-N Gallery"]
    for row in select:
        gallery.append(
         f"## {row['title']} · seed {row['seed']}\n\n"
         f"```text\n{row['body']}\n```"
        )
    (out_dir / "gallery.md").write_text(
    "\n\n".join(gallery), encoding = "UTF-8"
    )

    print("selected", len(select), "/", len(TITLES))
    print("missing", len(summary['missing']))
    return summary

if __name__ == "__main__":
    smoke = "--smoke" in sys.argv
    best_of_n(
        smoke = smoke,
        n_per_titles = 1 if smoke else 30
    )