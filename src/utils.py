"""Tiện ích dùng chung: seed, nhật ký thí nghiệm, chuẩn hoá + tách từ tiếng Việt, I/O."""
import csv
import json
import random
import re
import time
import unicodedata
import zlib
from datetime import datetime
from multiprocessing import Pool

import numpy as np

import config


# ----------------------------------------------------------------- tái lập
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def stable_hash(s):
    """Hàm băm ổn định cho gensim (hash() mặc định của Python đổi theo PYTHONHASHSEED)."""
    return zlib.crc32(str(s).encode("utf-8"))


class Timer:
    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        self.dt = time.time() - self.t0


# ------------------------------------------------------- nhật ký thí nghiệm
LOG_FIELDS = ["timestamp", "experiment", "name", "seed", "config", "metrics", "duration_s"]


def _py(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def log_run(experiment, name, seed, cfg, metrics, duration_s):
    """Ghi MỌI lượt chạy vào experiment_log.csv (ngày giờ, cấu hình, seed, kết quả)."""
    new = not config.LOG_CSV.exists()
    with open(config.LOG_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if new:
            w.writeheader()
        w.writerow({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "experiment": experiment, "name": name, "seed": seed,
            "config": json.dumps(cfg, ensure_ascii=False, default=_py),
            "metrics": json.dumps(metrics, ensure_ascii=False, default=_py),
            "duration_s": round(float(duration_s), 2),
        })


# --------------------------------------------------------------- văn bản
_URL = re.compile(r"https?://\S+|www\.\S+")
# Chuẩn hoá kiểu bỏ dấu cũ -> mới ở cuối âm tiết: hoà->hòa, khoẻ->khỏe, thuỷ->thủy
# (không áp dụng sau 'q': quý giữ nguyên; không áp dụng khi sau còn chữ cái: ngoài, toán)
_TONE_RE = re.compile(
    r"(?<![qQ])[uU][ỳýỷỹỵ](?![^\W\d_])|[oO][àáảãạèéẻẽẹ](?![^\W\d_])")
_OA = dict(zip("àáảãạ", "òóỏõọ"))
_OE = dict(zip("èéẻẽẹ", "òóỏõọ"))
_UY = dict(zip("ỳýỷỹỵ", "ùúủũụ"))


def _fix_tone(m):
    s = m.group(0)
    a, b = s[0], s[1]
    al, bl = a.lower(), b.lower()
    if al == "o":
        new = (_OA[bl] + "a") if bl in _OA else (_OE[bl] + "e")
    else:
        new = _UY[bl] + "y"
    if a.isupper() and b.isupper():
        return new.upper()
    if a.isupper():
        return new[0].upper() + new[1:]
    return new


def normalize_text(s):
    """Chuẩn hoá Unicode (NFC), bỏ URL/ký tự vô hình, thống nhất vị trí dấu, gộp khoảng trắng."""
    s = unicodedata.normalize("NFC", str(s)).replace("\u200b", "")
    s = _URL.sub(" ", s)
    s = _TONE_RE.sub(_fix_tone, s)
    return re.sub(r"\s+", " ", s).strip()


def segment(text, tokenizer="pyvi"):
    """Tách từ tiếng Việt -> chuỗi có dấu gạch dưới nối âm tiết của từ ghép (giữ nguyên hoa/thường)."""
    if tokenizer == "pyvi":
        from pyvi import ViTokenizer
        return ViTokenizer.tokenize(text)
    if tokenizer == "underthesea":
        from underthesea import word_tokenize
        return word_tokenize(text, format="text")
    raise ValueError(tokenizer)


_WORD = re.compile(r"^[^\W_]+(?:_[^\W_]+)*$")


def to_tokens(segmented):
    """Chuỗi đã tách từ -> list token chữ thường; bỏ dấu câu; số thuần -> <num>."""
    out = []
    for t in segmented.lower().split():
        if t.isdigit():
            out.append("<num>")
        elif _WORD.match(t):
            out.append(t)
    return out


_TOK = None


def _init(tokenizer):
    global _TOK
    _TOK = tokenizer


def _seg_one(text):
    return segment(text, _TOK)


def segment_many(texts, tokenizer="pyvi", n_jobs=1, chunksize=256):
    if n_jobs <= 1:
        return [segment(t, tokenizer) for t in texts]
    with Pool(n_jobs, initializer=_init, initargs=(tokenizer,)) as p:
        return p.map(_seg_one, texts, chunksize=chunksize)


# --------------------------------------------------------------------- I/O
def write_lines(path, lines):
    with open(path, "w", encoding="utf-8") as f:
        for l in lines:
            f.write(l + "\n")


def read_lines(path):
    with open(path, encoding="utf-8") as f:
        return [l.rstrip("\n") for l in f]


def load_token_lines(path, max_lines=None):
    out = []
    with open(path, encoding="utf-8") as f:
        for i, l in enumerate(f):
            if max_lines is not None and i >= max_lines:
                break
            out.append(l.split())
    return out


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def md_table(header, rows):
    out = ["| " + " | ".join(map(str, header)) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(map(str, r)) + " |")
    return "\n".join(out)
