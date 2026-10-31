import pickle
import torch
from pathlib import Path
import numpy as np
import math
import csv
import time
from collections import deque

from model import config, Poetry_Cloud
from tokenizer import encoder, decoder

class Data_lr:
    max_lr:float = 8e-5
    min_lr:float = 8e-6
    warmup_iter:int = 50
    max_step: int = 1000

ROOT = Path(__file__).resolve().parent

OUT_DIR = ROOT / "out" / "sft"
OUT_DIR.mkdir(parents = True, exist_ok = True)
DATA_DIR = ROOT / "data" / "poetry_sft_v1"
METRICS_CSV = OUT_DIR / "metrics.csv"
SAMPLE_DIR = OUT_DIR / "samples"

LOG_INTERVAL = 20
VAL_INTERVAL = 100
SAVE_INTERVAL = 250
SAMPLE_INTERVAL = 250

SAMPLE_DIR.mkdir(parents = True, exist_ok = True)

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

block_size = 128
batch_size = 32
device = "cuda" if torch.cuda.is_available() else "cpu"
cfg = config(
    vocab_size = meta["vocab_size"],
    block_size = block_size
)

poetry_cloud = Poetry_Cloud(cfg).to(device)

SAMPLE_PROMPTS = [
    ("<|律诗|>", "溫泉"),
    ("<|绝句|>", "秋夜"),
    ("<|绝句|>", "送別"),
    ("<|律诗|>", "山行"),
    ("<|绝句|>", "江上"),
]

def get_batch(split):
    data = train_data if split == "train" else val_data

    ix = torch.randint(
        len(data) - block_size,
        (batch_size,)
    ).tolist()

    x = torch.stack([
        torch.from_numpy(data[i : i + block_size].astype(np.int64))
        for i in ix
    ])

    y = torch.stack([
        torch.from_numpy(data[i + 1: i + 1 + block_size].astype(np.int64))
        for i in ix
    ])

    return x.to(device), y.to(device)

@torch.no_grad
def estimate_loss():
    poetry_cloud.eval()
    losses = []
    for _ in range(20):
        x, y = get_batch("val")
        logits, loss = poetry_cloud(x, y)
        losses.append(loss.item())

    return sum(losses) / len(losses)

def get_lr(step, data_lr):
    if step < data_lr.warmup_iter:
        return data_lr.max_lr * (step + 1) / data_lr.warmup_iter

    if step >= data_lr.max_step:
        return data_lr.min_lr

    progress = (step - data_lr.warmup_iter) / (data_lr.max_step - data_lr.warmup_iter)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return data_lr.min_lr + coeff * (data_lr.max_lr - data_lr.min_lr)

def save_checkpoint(done_step, type = "normal", loss_value = None):
    ckpt_dir = OUT_DIR / "ckpt"
    ckpt_dir.mkdir(parents = True, exist_ok = True)
    torch.save({
        "model" : poetry_cloud.state_dict(),
        "optimizer" : optim.state_dict(),
        "model_args" : vars(cfg),
        "iter_step" : done_step,
        "val_loss" : loss_value,
        "best_loss" : best_loss
    }, OUT_DIR / "ckpt" / f"ckpt{type}.pt")

def append_metrics(step, lr, train_loss, val_loss, grad_norm, elapsed):
    new_file = not METRICS_CSV.exists()

    with METRICS_CSV.open("a", encoding = "UTF-8", newline = "") as f:
        writer = csv.writer(f)

        if new_file:
            writer.writerow([
                "step",
                "lr",
                "train_loss",
                "val_loss",
                "grad_norm",
                "elapsed_sec",
                "tokens_per_sec",
            ])

        tokens_per_sec = (
            (step + 1 - start_step) * batch_size * block_size / elapsed
            if elapsed > 0 else 0
        )

        writer.writerow([
            step,
            f"{lr:.3e}",
            round(train_loss, 4),
            "" if val_loss is None else round(val_loss, 4),
            round(grad_norm, 3),
            round(elapsed, 1),
            int(tokens_per_sec),
        ])

def save_sample_poems(step):
    poetry_cloud.eval()

    sections = [
        f"step = {step}",
        "temperature = 0.8",
        "top_k = 40",
        "max_tokens = 80",
        ""
    ]

    for index, (genre, title) in enumerate(SAMPLE_PROMPTS):
        prompt = f"{genre}{title}\n"

        prompt_idx = encoder(prompt, meta)

        x = torch.tensor(
            [prompt_idx],
            dtype = torch.long,
            device = device
        )

        torch.manual_seed(1234 + index)

        out = poetry_cloud.generate(
            x,
            max_tokens = 80,
            temperature = 0.8,
            top_k = 40,
            stop_id = meta["eot_id"]
        )

        new_idx = out[0, x.size(1):].tolist()
        text = decoder(new_idx, meta)
        sections.append(f"### {title}")
        sections.append(text)
        sections.append("")

    sample_path = SAMPLE_DIR / f"step_{step:06d}.txt"
    sample_path.write_text("\n".join(sections), encoding = "UTF-8")

    poetry_cloud.train()

if __name__ == "__main__":
    data_lr = Data_lr()
    start_step = 0
    poetry_cloud.train()
    best_loss = 10.0
    last_loss = 10.0

    decay = [
        p for p in poetry_cloud.parameters() if p.dim() >= 2
    ]

    no_decay = [
        p for p in poetry_cloud.parameters() if p.dim() < 2
    ]

    ckpt = torch.load(ROOT / "out" / "pretrain" / "ckpt" / "ckptbest.pt", map_location = device)
    poetry_cloud.load_state_dict(ckpt["model"])

    optim = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 0.1},
            {"params": no_decay, "weight_decay": 0.0}
        ],
        lr = data_lr.max_lr,
        betas = (0.9, 0.95)
    )

    start_time = time.time()
    recent_loss = deque(maxlen = LOG_INTERVAL)
    for step in range(start_step, data_lr.max_step):
        lr = get_lr(step, data_lr)

        for group in optim.param_groups:
            group["lr"] = lr

        poetry_cloud.train()
        optim.zero_grad(set_to_none = True)
        x, y = get_batch("train")
        with torch.autocast("cuda", dtype = torch.bfloat16, enabled = (device =="cuda")):
            logits, loss = poetry_cloud(x, y)
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            poetry_cloud.parameters(),
            1.0
        )
        optim.step()
        recent_loss.append(loss.item())

        if step % LOG_INTERVAL == 0:
            val_loss = None

            if step % VAL_INTERVAL == 0:
                val_loss = estimate_loss()
                last_loss = val_loss
                print(f"step = {step},"
                      f"train_loss = {sum(recent_loss) / len(recent_loss)},"
                      f"val_loss = {val_loss}")
                if val_loss < best_loss:
                    best_loss = val_loss
                    save_checkpoint(step, "best", best_loss)

            append_metrics(
                step = step,
                lr = lr,
                train_loss = sum(recent_loss) / len(recent_loss),
                val_loss = val_loss,
                grad_norm = grad_norm.item(),
                elapsed = time.time() - start_time
            )

        if step % SAVE_INTERVAL == 0:
            save_checkpoint(step, loss_value = last_loss)

        if step % SAMPLE_INTERVAL == 0:
            save_sample_poems(step)

    save_checkpoint(data_lr.max_step, loss_value = last_loss)