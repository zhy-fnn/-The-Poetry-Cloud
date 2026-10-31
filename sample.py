import pickle
import sys
from pathlib import Path
import torch

from model import config, Poetry_Cloud
from tokenizer import encoder, decoder

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "poetry_char_v3"
CKPT = ROOT / "out" / "sft" / "ckpt" / "ckptbest.pt"

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_meta(meta_path = DATA_DIR / "meta.pkl"):
    with meta_path.open("rb") as f:
        meta = pickle.load(f)
    return meta

def load_model(ckpt_path = CKPT, meta_path = None):
    ck = torch.load(ckpt_path, map_location = device)
    model = Poetry_Cloud(config(**ck["model_args"])).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck

def make_poem(model, meta, title, genre, temperature = 0.8, top_k = 40, max_tokens = 80, seed = None):
    if seed is not None:
        torch.manual_seed(seed)

    ids = torch.tensor(
        [encoder(f"{genre}{title}\n", meta)],
        dtype = torch.long,
        device = device,
    )

    out = model.generate(
        ids,
        max_tokens = max_tokens,
        temperature = temperature,
        top_k = top_k,
        stop_id = meta["eot_id"],
    )

    return decoder(out[0, ids.size(1):].tolist(), meta)

def sample_p():
    meta = load_meta()
    model, ck = load_model()
    title = sys.argv[1] if len(sys.argv) > 1 else "溫泉"
    genre = sys.argv[2] if len(sys.argv) > 2 else "<|绝句|>"
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    count = int(sys.argv[4]) if len(sys.argv) > 4 else 3

    for i in range(count):
        print(f"\n--- {title} 第 {i + 1} 首 ---")
        print(make_poem(model, meta, title, genre, seed = seed + i))

if __name__ == "__main__":
    sample_p()