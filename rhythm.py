import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
YUN_PATH = ROOT / "data" / "poetry_char_v3" / "yun_trad.json"

with YUN_PATH.open("r", encoding = "UTF-8") as f:
    YUN = json.load(f)

C2R = {}
for groups, chars in YUN.items():
    for ch in chars:
        C2R.setdefault(ch, set()).add(groups)

def last_cjk(text : str):
    chars = re.findall(r"[\u4e00-\u9fff]", text)
    return chars[-1] if chars else None

def first_line_last_cjk(paragraph):
    head = re.split("[，。？！；、]", paragraph, maxsplit = 1)[0]
    return last_cjk(head)

def rhythm_ok(paragraph):
    if len(paragraph) not in (2, 4):
        return None

    required = [last_cjk(p) for p in paragraph]
    if any(c is None for c in required):
        return None

    groups = [C2R.get(c) for c in required]
    if any(g is None for g in groups):
        return None

    common = set.intersection(*groups)
    return bool(common)