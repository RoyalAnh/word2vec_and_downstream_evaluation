"""Bước 2 - Huấn luyện word2vec (gensim) ở cả hai chế độ Skip-gram và CBOW.

Lưới: {sg, cbow} x window {3, 8} x dim {100, 300} = 8 mô hình, cùng ngữ liệu, cùng seed.
Mặc định workers=1 để kết quả tái lập tuyệt đối (gensim chỉ tất định khi 1 luồng); dùng --workers 4 để nhanh hơn
(khi đó kết quả có thể lệch rất nhỏ giữa các lần chạy - phải ghi rõ trong báo cáo).
"""
import argparse

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

import config
from utils import Timer, log_run, set_seed, stable_hash


def train_w2v(sentences, mode, window, dim, seed, workers, epochs, min_count=config.MIN_COUNT):
    return Word2Vec(
        sentences=sentences, vector_size=dim, window=window, min_count=min_count,
        sg=1 if mode == "sg" else 0, hs=0, negative=config.W2V_NEGATIVE, sample=1e-3,
        epochs=epochs, workers=workers, seed=seed, hashfxn=stable_hash)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=config.W2V_EPOCHS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--only", default=None, help="chỉ huấn luyện mô hình có tên này")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    corpus = config.DATA_DIR / "wiki_tokens.txt"
    for mode in config.W2V_MODES:
        for window in config.W2V_WINDOWS:
            for dim in config.W2V_DIMS:
                name = config.w2v_name(mode, window, dim)
                if args.only and args.only != name:
                    continue
                out = config.MODEL_DIR / f"{name}.kv"
                if out.exists() and not args.force:
                    print(f"[skip] {name} đã có")
                    continue
                set_seed(args.seed)
                with Timer() as t:
                    model = train_w2v(LineSentence(str(corpus)), mode, window, dim, args.seed,
                                      args.workers, args.epochs)
                model.wv.save(str(out))
                cfg = dict(mode=mode, window=window, dim=dim, epochs=args.epochs, workers=args.workers,
                           min_count=config.MIN_COUNT, negative=config.W2V_NEGATIVE, sample=1e-3)
                met = dict(vocab_size=len(model.wv), corpus_words=int(model.corpus_total_words),
                           corpus_sentences=int(model.corpus_count))
                log_run("train_w2v", name, args.seed, cfg, met, t.dt)
                print(f"[done] {name}: vocab={met['vocab_size']:,} time={t.dt:.0f}s")


if __name__ == "__main__":
    main()
