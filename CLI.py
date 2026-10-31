import argparse
import json
import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GENRE_TAG = {"绝句" : "<|绝句|>", "律诗" : "<|律诗|>"}

def cmd_sample(arg):
    from sample import load_model, load_meta, make_poem

    meta = load_meta()
    model, _ = load_model(arg.ckpt)
    genre = GENRE_TAG[arg.genre]

    for i in range(arg.count):
        body = make_poem(
            model, meta, arg.prompt, genre, seed = arg.seed + i
        )
        print(f"--- {arg.prompt} 第 {i + 1} 首 ---\n{body}")

def cmd_train(arg):
    data = Path(arg.data)
    out = Path(arg.out)

    required = ("meta.pkl", "val.bin", "train.bin", "sft_manifest.json")
    missing = [name for name in required if not (data / name).is_file()]
    if missing:
        raise SystemExit(f"[dry-run]缺文件：{missing}")

    empty = [name for name in required if (data / name).stat().st_size == 0]
    if empty:
        raise SystemExit(f"[dry-run]空文件：{empty}")

    with (data / "meta.pkl").open("rb") as f:
        meta = pickle.load(f)

    manifest = json.loads(
        (data / "sft_manifest.json").read_text(encoding = "UTF-8")
    )

    train_bytes = (data / "train.bin").stat().st_size
    val_bytes = (data / "val.bin").stat().st_size
    init_ckpt = (ROOT / "out" / "pretrain" / "ckpt" / "ckptbest.pt")

    check = {
        "meta_vocab" : (
                len(meta["itos"]) == meta["vocab_size"] and len(meta["stoi"]) == meta["vocab_size"]
        ),
        "special_ids" : "unk_id" in meta and "eot_id" in meta,
        "train_even_bytes" : train_bytes % 2 == 0,
        "val_even_bytes" : val_bytes % 2 == 0,
        "train_tokens" : train_bytes // 2 == manifest["train_tokens"],
        "val_tokens" : val_bytes // 2 == manifest["val_tokens"],
        "split_total" : (
                manifest["train_data"] + manifest["val_data"] == manifest["total"]
        ),
        "genre_total" : sum(manifest["counts"].values()) == manifest["total"],
        "init_ckpt" : init_ckpt.is_file() and init_ckpt.stat().st_size > 0
    }

    print("[dry-run] data:", data)
    print("[dry-run] out:", out)
    print("[dry-run] vocab:", meta["vocab_size"])
    print("[dry-run] train:", manifest["train_data"], "首 /", manifest["train_tokens"], "tokens")
    print("[dry-run] val:", manifest["val_data"], "首 /", manifest["val_tokens"], "tokens")
    print("[dry-run] init ckpt:", init_ckpt)

    fail = [name for name, ok in check.items() if ok is not True]
    if fail:
        raise SystemExit(f"检查失败：{fail}")
    print("[dry-run] all passed")

    if not arg.dry_run:
        raise SystemExit("完整训练尚未接入：当前使用 python train_sft.py")

def best_of_n(arg):
    from best_of_N import best_of_n

    summary = best_of_n(
        ckpt = arg.ckpt,
        n_per_titles = arg.n_per_titles,
        smoke = arg.smoke,
        out_dir = arg.out_dir
    )

    if summary is not None:
        print("missing：", summary["missing"])

def build_parser():
    parse = argparse.ArgumentParser(prog = "cli", description = "诗云的命令行入口")
    parse.add_argument("--version", action = "version", version = "v0.1")
    sub = parse.add_subparsers(dest = "cmd", required = True)

    p = sub.add_parser("sample", help = "用ckpt生成诗")
    p.add_argument("--ckpt", type = Path, default = ROOT / "out" / "sft" / "ckpt" / "ckptbest.pt")
    p.add_argument("--prompt", required = True)
    p.add_argument("--genre", choices = ["绝句", "律诗"], default = "绝句")
    p.add_argument("--seed", type = int, default = 1000)
    p.add_argument("--count", type = int, default = 1)
    p.set_defaults(func = cmd_sample)

    p = sub.add_parser("train", help = "打印训练配置")
    p.add_argument("--data", type = Path, default = ROOT / "data" / "poetry_sft_v1")
    p.add_argument("--out", type = Path, default = ROOT / "out" / "sft")
    p.add_argument("--dry-run", action = "store_true")
    p.set_defaults(func = cmd_train)

    p = sub.add_parser("best_of_n", help = "生成n个诗中最好的")
    p.add_argument("--ckpt", type = Path, default = ROOT / "out" / "pretrain" / "ckpt" / "ckptbest.pt")
    p.add_argument("--n_per_titles", type = int, default = 30)
    p.add_argument("--smoke", action = "store_true")
    p.add_argument("--out_dir", type = Path, default = ROOT / "out" / "best_of_n")
    p.set_defaults(func = best_of_n)
    return parse

def main(argv = None):
    args = build_parser().parse_args(argv)
    args.func(args)

if __name__ == "__main__":
    main()