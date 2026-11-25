"""Bước 8 - Ablation về kích thước và miền của ngữ liệu (trả lời câu hỏi "khi nào embedding tự huấn luyện thua").

Cố định MỘT cấu hình word2vec (--cfg, ví dụ sg_w3_d300) và đổi ngữ liệu huấn luyện:
  wiki_10 / wiki_30 / wiki_100 (% số câu của ngữ liệu Wikipedia, đã xáo trộn sẵn),
  vsfc_only (chỉ câu của UIT-VSFC train - cùng miền nhưng rất nhỏ), wiki_100+vsfc (gộp).
Mỗi biến thể: huấn luyện w2v -> embedding trung bình + MLP (nhiều seed) + độ chính xác analogy.
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config
from eval_extrinsic import eval_mean_mlp, load_vsfc_data
from eval_intrinsic import analogy_eval, build_questions
from train_w2v import train_w2v
from utils import Timer, load_token_lines, log_run, read_jsonl, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="sg_w3_d300", help="dạng {sg|cbow}_w{window}_d{dim}")
    ap.add_argument("--task", default="sentiment", choices=["sentiment", "topic"])
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=config.W2V_EPOCHS)
    ap.add_argument("--seeds", type=int, nargs="*", default=config.SEEDS)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    mode, w, d = args.cfg.split("_")
    window, dim = int(w[1:]), int(d[1:])

    wiki = load_token_lines(config.DATA_DIR / "wiki_tokens.txt")
    vs_train = [r["tok"].split() for r in read_jsonl(config.DATA_DIR / "vsfc_train.jsonl")]
    variants = {
        "wiki_10": wiki[: int(0.1 * len(wiki))],
        "wiki_30": wiki[: int(0.3 * len(wiki))],
        "wiki_100": wiki,
        "vsfc_only": vs_train,
        "wiki_100+vsfc": wiki + vs_train,
    }
    data = load_vsfc_data(args.task)
    questions = build_questions()
    apath = config.RESULT_DIR / "analogy_vi.txt"
    summary = []
    for vname, sents in variants.items():
        n_tok = sum(len(s) for s in sents)
        set_seed(args.seed)
        with Timer() as t:
            model = train_w2v(sents, mode, window, dim, args.seed, args.workers, args.epochs)
        kv = model.wv
        kv.save(str(config.MODEL_DIR / "ablation" / f"{args.cfg}__{vname}.kv"))
        intr = analogy_eval(kv, questions, apath)
        res = eval_mean_mlp(kv, data, f"{args.cfg}__{vname}", args.seeds, args.task,
                            experiment="corpus_ablation_mlp")
        f1s = [r["test_macro_f1"] for r in res]
        met = {"n_sentences": len(sents), "n_tokens": n_tok, "vocab_size": len(kv.key_to_index),
               "analogy_acc": intr["analogy_acc"], "analogy_covered": intr["analogy_covered"],
               "mlp_macro_f1_mean": float(np.mean(f1s)),
               "mlp_macro_f1_std": float(np.std(f1s, ddof=1)) if len(f1s) > 1 else 0.0}
        log_run("corpus_ablation", f"{args.cfg}__{vname}", args.seed, {"task": args.task, "cfg": args.cfg}, met, t.dt)
        summary.append((vname, n_tok, met["mlp_macro_f1_mean"], met["mlp_macro_f1_std"]))
        print(f"[{vname}] tokens={n_tok:,} vocab={met['vocab_size']:,} analogy={met['analogy_acc']:.3f} "
              f"MLP macro-F1={met['mlp_macro_f1_mean']:.4f} ± {met['mlp_macro_f1_std']:.4f}")

    fig, ax = plt.subplots(figsize=(6, 4))
    wk = [s for s in summary if s[0].startswith("wiki_") and "+" not in s[0]]
    ax.errorbar([s[1] for s in wk], [s[2] for s in wk], yerr=[s[3] for s in wk], marker="o", capsize=3,
                label="Wikipedia 10/30/100%")
    for s in summary:
        if s[0] in ("vsfc_only", "wiki_100+vsfc"):
            ax.errorbar([s[1]], [s[2]], yerr=[s[3]], marker="s", capsize=3, label=s[0])
    ax.set_xscale("log")
    ax.set_xlabel("số token huấn luyện embedding")
    ax.set_ylabel("macro-F1 (test)")
    ax.set_title(f"Ảnh hưởng ngữ liệu - {args.cfg}")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "corpus_ablation.png", dpi=150)


if __name__ == "__main__":
    main()
