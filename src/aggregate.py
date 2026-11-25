"""Bước 9 - Tổng hợp nhật ký thí nghiệm thành bảng (Markdown/CSV) và biểu đồ cho báo cáo.
Mọi con số lấy TRỰC TIẾP từ results/experiment_log.csv (để số liệu báo cáo khớp log).
Nếu một cấu hình được chạy lại, chỉ lượt chạy MỚI NHẤT của (experiment, name, seed, task) được dùng.
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from utils import md_table


def load_log():
    df = pd.read_csv(config.LOG_CSV)
    df["config"] = df["config"].apply(json.loads)
    df["metrics"] = df["metrics"].apply(json.loads)
    df["task"] = df["config"].apply(lambda c: c.get("task", ""))
    df = df.sort_values("timestamp").drop_duplicates(["experiment", "name", "seed", "task"], keep="last")
    return df


def sel(df, exp, task=None):
    d = df[df.experiment == exp]
    if task is not None:
        d = d[d.task.isin([task, ""])]
    return d


def agg(d, keys=("test_acc", "test_macro_f1", "dev_macro_f1")):
    rows = []
    for name, g in d.groupby("name"):
        row = {"name": name, "n_runs": len(g)}
        for k in keys:
            v = [m[k] for m in g.metrics if m.get(k) is not None]
            row[k] = float(np.mean(v)) if v else np.nan
            row[k + "_std"] = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def fmt(m, s):
    return "-" if pd.isna(m) else f"{m:.4f} ± {s:.4f}"


def table_md(df, cols):
    rows = [[r["name"], r["n_runs"]] + [fmt(r[c], r[c + "_std"]) for c in cols] for _, r in df.iterrows()]
    return md_table(["cấu hình", "số lượt"] + cols, rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="sentiment")
    args = ap.parse_args()
    df = load_log()
    out = [f"# Tổng hợp kết quả (task = {args.task})\n",
           "Giá trị: trung bình ± độ lệch chuẩn mẫu (ddof=1) trên các seed; lượt chạy xác định (1 lượt) có ± 0.\n"]

    # --- Bảng 1: lưới word2vec/SVD/GloVe: nội tại + ngoại tại
    mlp = agg(sel(df, "extrinsic_mean_mlp", args.task))
    intr = sel(df, "intrinsic").set_index("name")
    if len(mlp):
        mlp["analogy_acc"] = mlp["name"].map(lambda n: intr.loc[n, "metrics"].get("analogy_acc", np.nan) if n in intr.index else np.nan)
        mlp["analogy_cov"] = mlp["name"].map(lambda n: f"{intr.loc[n, 'metrics'].get('analogy_covered', 0)}/{intr.loc[n, 'metrics'].get('analogy_total', 0)}" if n in intr.index else "-")
        mlp = mlp.sort_values("dev_macro_f1", ascending=False)
        rows = [[r["name"], f"{r['analogy_acc']:.3f}" if not pd.isna(r["analogy_acc"]) else "-", r["analogy_cov"],
                 fmt(r["test_acc"], r["test_acc_std"]), fmt(r["test_macro_f1"], r["test_macro_f1_std"])]
                for _, r in mlp.iterrows()]
        t1 = md_table(["mô hình embedding", "analogy acc", "analogy phủ", "MLP test acc", "MLP test macro-F1"], rows)
        out += ["## Bảng 1. Nội tại (analogy) và ngoại tại (embedding trung bình + MLP)\n", t1, ""]
        (config.TABLE_DIR / f"table1_embedding_grid_{args.task}.md").write_text(t1, encoding="utf-8")

    # --- Bảng 2: so sánh 4 cách biểu diễn (chọn cấu hình tốt nhất theo DEV macro-F1)
    cands = []
    for exp, label in [("extrinsic_linear", None), ("extrinsic_mean_mlp", "Embedding TB + MLP (tốt nhất)"),
                       ("extrinsic_neural", "Embedding + CNN/LSTM (tốt nhất)"),
                       ("extrinsic_phobert", None)]:
        a = agg(sel(df, exp, args.task))
        if not len(a):
            continue
        if label is None:
            for _, r in a.iterrows():
                cands.append((r["name"], r))
        else:
            best = a.sort_values("dev_macro_f1", ascending=False).iloc[0]
            cands.append((f"{label}: {best['name']}", best))
    if cands:
        rows = [[n, r["n_runs"], fmt(r["test_acc"], r["test_acc_std"]), fmt(r["test_macro_f1"], r["test_macro_f1_std"])]
                for n, r in cands]
        t2 = md_table(["biểu diễn / mô hình", "số lượt", "test acc", "test macro-F1"], rows)
        out += ["## Bảng 2. So sánh các cách biểu diễn (chọn theo dev)\n", t2, ""]
        (config.TABLE_DIR / f"table2_representations_{args.task}.md").write_text(t2, encoding="utf-8")
        fig, ax = plt.subplots(figsize=(8, 0.5 * len(cands) + 1.5))
        ax.barh([n for n, _ in cands][::-1], [r["test_macro_f1"] for _, r in cands][::-1],
                xerr=[r["test_macro_f1_std"] for _, r in cands][::-1], capsize=3)
        ax.set_xlabel("test macro-F1")
        ax.tick_params(axis="y", labelsize=7)
        fig.tight_layout()
        fig.savefig(config.FIG_DIR / f"representations_compare_{args.task}.png", dpi=150)
        plt.close(fig)

    # --- Bảng 3: CNN/LSTM ablation
    neu = agg(sel(df, "extrinsic_neural", args.task), keys=("test_acc", "test_macro_f1", "dev_macro_f1", "best_epoch", "trainable_params"))
    if len(neu):
        t3 = table_md(neu.sort_values("name"), ["test_macro_f1", "test_acc", "trainable_params"])
        out += ["## Bảng 3. TextCNN/BiLSTM: khởi tạo ngẫu nhiên vs word2vec (đóng băng / fine-tune)\n", t3, ""]
        (config.TABLE_DIR / f"table3_neural_{args.task}.md").write_text(t3, encoding="utf-8")

    # --- Bảng 4: ngữ liệu
    ab = sel(df, "corpus_ablation")
    if len(ab):
        rows = [[r["name"], f"{r['metrics']['n_tokens']:,}", f"{r['metrics']['vocab_size']:,}",
                 f"{r['metrics']['analogy_acc']:.3f}",
                 f"{r['metrics']['mlp_macro_f1_mean']:.4f} ± {r['metrics']['mlp_macro_f1_std']:.4f}"]
                for _, r in ab.iterrows()]
        t4 = md_table(["biến thể ngữ liệu", "số token", "từ vựng", "analogy acc", "MLP macro-F1"], rows)
        out += ["## Bảng 4. Ảnh hưởng kích thước/miền ngữ liệu\n", t4, ""]
        (config.TABLE_DIR / f"table4_corpus_{args.task}.md").write_text(t4, encoding="utf-8")

    # --- Bảng 5: thiên kiến
    bw = sel(df, "bias_weat")
    if len(bw):
        rows = [[r["name"], f"{r['metrics'].get('effect_size', float('nan')):+.3f}",
                 f"{r['metrics'].get('p_value', float('nan')):.4f}",
                 ", ".join(r["metrics"]["most_male"]), ", ".join(r["metrics"]["most_female"])]
                for _, r in bw.iterrows()]
        t5 = md_table(["mô hình", "WEAT effect size", "p", "5 nghề nghiêng nam nhất", "5 nghề nghiêng nữ nhất"], rows)
        out += ["## Bảng 5. Thiên kiến giới\n", t5, ""]
        (config.TABLE_DIR / f"table5_bias_{args.task}.md").write_text(t5, encoding="utf-8")

    # --- Bảng 6: thời gian huấn luyện embedding
    tr = df[df.experiment.isin(["train_w2v", "train_svd", "train_glove"])]
    if len(tr):
        rows = [[r["name"], r["experiment"], f"{r['duration_s']:.0f}s"] for _, r in tr.iterrows()]
        t6 = md_table(["mô hình", "loại", "thời gian"], rows)
        out += ["## Bảng 6. Thời gian huấn luyện embedding\n", t6, ""]
        (config.TABLE_DIR / f"table6_train_time_{args.task}.md").write_text(t6, encoding="utf-8")

    (config.RESULT_DIR / f"summary_{args.task}.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
