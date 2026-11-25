"""Bước 1 - Chuẩn bị dữ liệu.

(a) Ngữ liệu huấn luyện embedding: Wikipedia tiếng Việt (HF `wikimedia/wikipedia`, 20231101.vi, streaming,
    lấy N bài ngẫu nhiên có seed) hoặc file văn bản cục bộ (--wiki_txt).
(b) Bộ dữ liệu hạ nguồn: UIT-VSFC (HF `uitnlp/vietnamese_students_feedback` hoặc thư mục cục bộ --vsfc_dir).
Các bước: chuẩn hoá Unicode -> tách câu -> tách từ -> token chữ thường -> thống kê + luật Zipf.
"""
import argparse
import collections
import json
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config
from utils import (Timer, log_run, md_table, normalize_text, segment_many, set_seed, to_tokens,
                   write_jsonl, write_lines)


def split_sentences(text):
    return [p.strip() for p in re.split(r"(?<=[.!?…])\s+|\n+", text) if p.strip()]


def collect_wiki(args):
    paragraphs = []
    if args.wiki_txt:
        p = Path(args.wiki_txt)
        files = sorted(p.glob("*.txt")) if p.is_dir() else [p]
        for fp in files:
            with open(fp, encoding="utf-8") as f:
                paragraphs.extend(l for l in f if l.strip())
    else:
        from datasets import load_dataset
        ds = load_dataset("wikimedia/wikipedia", "20231101.vi", split="train", streaming=True)
        ds = ds.shuffle(seed=args.seed, buffer_size=10_000)
        for i, ex in enumerate(ds):
            if i >= args.n_articles:
                break
            paragraphs.append(ex["text"])
    sents = []
    for para in paragraphs:
        for s in split_sentences(normalize_text_keep_lines(para)):
            n = len(s.split())
            if 5 <= n <= 100:
                sents.append(s)
        if len(sents) >= args.max_sentences:
            break
    sents = list(dict.fromkeys(sents))[: args.max_sentences]   # bỏ trùng, giữ thứ tự
    return sents


def normalize_text_keep_lines(text):
    """Chuẩn hoá từng dòng (giữ ký tự xuống dòng để còn tách câu)."""
    return "\n".join(normalize_text(l) for l in text.split("\n"))


def load_vsfc(vsfc_dir):
    """Trả về {'train':[{sentence,sentiment,topic}], 'dev':..., 'test':...}.
    Hỗ trợ: (1) thư mục gốc UIT-VSFC (train|dev|test/{sentences,sentiments,topics}.txt),
            (2) thư mục có train.csv/dev.csv/test.csv (cột sentence,sentiment,topic),
            (3) tải qua Hugging Face datasets."""
    out = {}
    if vsfc_dir:
        d = Path(vsfc_dir)
        for split in ("train", "dev", "test"):
            sd = d / split
            if (sd / "sentences.txt").exists():
                sents = [l.rstrip("\n") for l in open(sd / "sentences.txt", encoding="utf-8")]
                sen = [int(l) for l in open(sd / "sentiments.txt", encoding="utf-8") if l.strip()]
                top = [int(l) for l in open(sd / "topics.txt", encoding="utf-8") if l.strip()]
                out[split] = [{"sentence": a, "sentiment": b, "topic": c} for a, b, c in zip(sents, sen, top)]
            elif (d / f"{split}.csv").exists():
                import pandas as pd
                df = pd.read_csv(d / f"{split}.csv")
                out[split] = df[["sentence", "sentiment", "topic"]].to_dict("records")
            else:
                raise FileNotFoundError(f"Không thấy dữ liệu VSFC cho split '{split}' trong {d}")
        return out
    from datasets import load_dataset
    ds = load_dataset("uitnlp/vietnamese_students_feedback")
    mapping = {"train": "train", "dev": "validation", "test": "test"}
    for split, hf in mapping.items():
        out[split] = [{"sentence": r["sentence"], "sentiment": int(r["sentiment"]), "topic": int(r["topic"])}
                      for r in ds[hf]]
    return out


def zipf_stats(counter, path_png, title):
    freqs = np.array(sorted(counter.values(), reverse=True), dtype=float)
    ranks = np.arange(1, len(freqs) + 1)
    lo, hi = 10, min(10000, len(freqs))
    slope = float("nan")
    if hi - lo > 5:
        slope, intercept = np.polyfit(np.log(ranks[lo:hi]), np.log(freqs[lo:hi]), 1)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.loglog(ranks, freqs, ".", ms=2, label="dữ liệu")
    if hi - lo > 5:
        ax.loglog(ranks[lo:hi], np.exp(intercept) * ranks[lo:hi] ** slope, "r-", label=f"fit, độ dốc={slope:.2f}")
    ax.set_xlabel("hạng (rank)")
    ax.set_ylabel("tần suất")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path_png, dpi=150)
    plt.close(fig)
    return float(slope)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_articles", type=int, default=30000)
    ap.add_argument("--max_sentences", type=int, default=1_500_000)
    ap.add_argument("--wiki_txt", default=None, help="file/thư mục .txt cục bộ thay cho Wikipedia HF")
    ap.add_argument("--vsfc_dir", default=None)
    ap.add_argument("--tokenizer", default="pyvi", choices=["pyvi", "underthesea"])
    ap.add_argument("--n_jobs", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true", help="dữ liệu giả để test đường ống")
    args = ap.parse_args()
    set_seed(args.seed)

    with Timer() as t:
        if args.smoke:
            import smoke_data
            raw = smoke_data.make_corpus()
            vsfc = smoke_data.make_vsfc()
            n_jobs = 1
        else:
            raw = collect_wiki(args)
            vsfc = load_vsfc(args.vsfc_dir)
            n_jobs = args.n_jobs
        print(f"[wiki] {len(raw):,} câu thô sau lọc")
        write_lines(config.DATA_DIR / "wiki_sentences.txt", raw)

        seg = segment_many(raw, args.tokenizer, n_jobs)
        tok_lines = [to_tokens(s) for s in seg]
        tok_lines = [t_ for t_ in tok_lines if len(t_) >= 3]
        write_lines(config.DATA_DIR / "wiki_tokens.txt", [" ".join(x) for x in tok_lines])

        cnt = collections.Counter(w for l in tok_lines for w in l)
        n_tokens = sum(cnt.values())
        vocab5 = {w for w, c in cnt.items() if c >= config.MIN_COUNT}
        slope = zipf_stats(cnt, config.FIG_DIR / "zipf_wiki.png", "Luật Zipf - ngữ liệu Wikipedia vi")
        lens = np.array([len(l) for l in tok_lines])
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        ax.hist(lens, bins=50)
        ax.set_xlabel("số token / câu")
        ax.set_ylabel("số câu")
        ax.set_title("Phân bố độ dài câu - ngữ liệu")
        fig.tight_layout()
        fig.savefig(config.FIG_DIR / "sentence_length_wiki.png", dpi=150)
        plt.close(fig)

        stats = {"corpus": {
            "n_sentences": len(tok_lines), "n_tokens": n_tokens, "vocab_all": len(cnt),
            "vocab_min_count": len(vocab5), "hapax_ratio": sum(1 for c in cnt.values() if c == 1) / len(cnt),
            "mean_len": float(lens.mean()), "zipf_slope": slope, "tokenizer": args.tokenizer}}

        # ---- UIT-VSFC
        vsfc_tokens = {}
        for split, rows in vsfc.items():
            texts = [normalize_text(r["sentence"]) for r in rows]
            sg = segment_many(texts, args.tokenizer, n_jobs)
            out_rows = []
            for r, tx, s in zip(rows, texts, sg):
                toks = to_tokens(s)
                out_rows.append({"text": tx, "seg": s, "tok": " ".join(toks),
                                 "sentiment": int(r["sentiment"]), "topic": int(r["topic"])})
            write_jsonl(config.DATA_DIR / f"vsfc_{split}.jsonl", out_rows)
            vsfc_tokens[split] = [r["tok"].split() for r in out_rows]
            stats[f"vsfc_{split}"] = {
                "n": len(out_rows),
                "mean_len": float(np.mean([len(x) for x in vsfc_tokens[split]])),
                "sentiment_dist": dict(collections.Counter(r["sentiment"] for r in out_rows)),
                "topic_dist": dict(collections.Counter(r["topic"] for r in out_rows))}
        vs_train_vocab = {w for l in vsfc_tokens["train"] for w in l}
        for split in ("dev", "test"):
            toks = [w for l in vsfc_tokens[split] for w in l]
            stats[f"vsfc_{split}"]["oov_vs_wiki_vocab"] = float(np.mean([w not in vocab5 for w in toks]))
            stats[f"vsfc_{split}"]["oov_vs_vsfc_train_vocab"] = float(np.mean([w not in vs_train_vocab for w in toks]))
        zipf_stats(collections.Counter(w for l in vsfc_tokens["train"] for w in l),
                   config.FIG_DIR / "zipf_vsfc.png", "Luật Zipf - UIT-VSFC (train)")

    with open(config.RESULT_DIR / "data_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    rows = [[k, json.dumps(v, ensure_ascii=False)] for k, v in stats.items()]
    (config.TABLE_DIR / "data_stats.md").write_text(md_table(["phần", "thống kê"], rows), encoding="utf-8")
    log_run("prepare_data", "corpus", args.seed, vars(args), stats, t.dt)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
