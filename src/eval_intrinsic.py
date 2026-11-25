"""Bước 3 - Đánh giá NỘI TẠI cho mọi mô hình *.kv (word2vec / SVD / GloVe):
  (1) láng giềng gần nhất cho >= 15 từ mẫu,
  (2) phép loại suy (analogy) tiếng Việt - bộ câu hỏi tự dựng từ các cặp từ bên dưới,
  (3) trực quan hoá PCA + t-SNE tô màu theo nhóm ngữ nghĩa.

LƯU Ý: các từ trong danh sách dưới đây phải KHỚP cách tách từ của pyvi/underthesea. Từ nào không có trong từ vựng
sẽ bị bỏ qua và được thống kê ở cột 'coverage'. Hãy xem lại danh sách sau khi chạy và bổ sung/sửa cho phù hợp.
"""
import argparse
import itertools

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from gensim.models import KeyedVectors

import config
from utils import Timer, log_run, md_table

SAMPLE_WORDS = [
    "học_sinh", "sinh_viên", "giáo_viên", "bác_sĩ", "bệnh_viện", "trường_học", "đại_học", "hà_nội",
    "việt_nam", "bóng_đá", "âm_nhạc", "máy_tính", "internet", "ngân_hàng", "tiền", "xe_máy", "ô_tô",
    "cơm", "phở", "mùa_xuân", "tình_yêu", "chiến_tranh", "hòa_bình", "kinh_tế", "chính_phủ", "thầy",
    "bố", "mẹ", "vua", "sông", "biển"]

GROUPS = {
    "quốc_gia": ["việt_nam", "thái_lan", "nhật_bản", "pháp", "đức", "nga", "trung_quốc", "hàn_quốc", "mỹ",
                 "ấn_độ", "lào", "campuchia", "úc"],
    "thủ_đô": ["hà_nội", "bangkok", "tokyo", "paris", "berlin", "moscow", "bắc_kinh", "seoul", "washington",
               "luân_đôn"],
    "gia_đình": ["bố", "mẹ", "anh", "chị", "em", "ông", "bà", "con_trai", "con_gái", "chồng", "vợ"],
    "nghề_nghiệp": ["bác_sĩ", "giáo_viên", "kỹ_sư", "luật_sư", "công_nhân", "nông_dân", "ca_sĩ", "nhà_báo",
                    "cảnh_sát", "diễn_viên"],
    "giáo_dục": ["học_sinh", "sinh_viên", "trường", "lớp", "thầy", "cô", "bài", "thi", "đại_học", "giáo_dục",
                 "học_tập"],
    "thể_thao": ["bóng_đá", "cầu_thủ", "huấn_luyện_viên", "trận", "giải", "bóng_rổ", "tennis", "đội_tuyển"],
    "ẩm_thực": ["cơm", "phở", "bún", "bánh", "thịt", "rau", "canh", "cà_phê"],
    "động_vật": ["chó", "mèo", "gà", "bò", "lợn", "cá", "chim", "voi", "hổ"],
    "thời_gian": ["ngày", "tháng", "năm", "tuần", "giờ", "sáng", "chiều", "tối", "mùa"],
    "màu_sắc": ["đỏ", "xanh", "vàng", "trắng", "đen", "tím", "hồng", "nâu"],
}

# Mỗi section là danh sách cặp (a, b); câu hỏi: a : b :: c : d với mọi cặp (a,b) != (c,d)
ANALOGY_PAIRS = {
    "thủ_đô-quốc_gia": [("hà_nội", "việt_nam"), ("bangkok", "thái_lan"), ("tokyo", "nhật_bản"),
                        ("paris", "pháp"), ("berlin", "đức"), ("moscow", "nga"), ("bắc_kinh", "trung_quốc"),
                        ("seoul", "hàn_quốc"), ("washington", "mỹ")],
    "giới_tính-gia_đình": [("ông", "bà"), ("anh", "chị"), ("bố", "mẹ"), ("nam", "nữ"),
                           ("con_trai", "con_gái"), ("chồng", "vợ"), ("đàn_ông", "phụ_nữ"), ("trai", "gái"),
                           ("thầy", "cô"), ("chú", "dì")],
    "trái_nghĩa": [("nóng", "lạnh"), ("cao", "thấp"), ("lớn", "nhỏ"), ("nhanh", "chậm"), ("tốt", "xấu"),
                   ("dài", "ngắn"), ("mạnh", "yếu"), ("giàu", "nghèo"), ("sáng", "tối"), ("vui", "buồn"),
                   ("trẻ", "già"), ("mới", "cũ")],
}


def build_questions():
    qs = {}
    for sec, pairs in ANALOGY_PAIRS.items():
        qs[sec] = [(a, b, c, d) for (a, b), (c, d) in itertools.permutations(pairs, 2)]
    return qs


def write_analogy_file(path, questions):
    with open(path, "w", encoding="utf-8") as f:
        for sec, qs in questions.items():
            f.write(f": {sec}\n")
            for q in qs:
                f.write(" ".join(q) + "\n")


def analogy_eval(kv, questions, path):
    n_total = sum(len(v) for v in questions.values())
    cov = {s: sum(all(w in kv.key_to_index for w in q) for q in qs) for s, qs in questions.items()}
    n_cov = sum(cov.values())
    res = {"analogy_acc": float("nan"), "analogy_covered": n_cov, "analogy_total": n_total, "per_section": {}}
    if n_cov == 0:
        return res
    # case_insensitive=False vì khoá từ vựng và câu hỏi đều đã là chữ thường
    score, sections = kv.evaluate_word_analogies(str(path), restrict_vocab=len(kv.key_to_index),
                                                 case_insensitive=False, dummy4unknown=False)
    res["analogy_acc"] = float(score)
    for s in sections:
        if s["section"].lower().startswith("total"):
            continue
        n = len(s["correct"]) + len(s["incorrect"])
        res["per_section"][s["section"]] = {"acc": (len(s["correct"]) / n) if n else float("nan"), "n": n,
                                            "coverage": cov.get(s["section"], 0) / len(questions[s["section"]])}
    return res


def neighbors_table(kv, words, topn=8):
    rows, found = [], 0
    for w in words:
        if w in kv.key_to_index:
            found += 1
            rows.append([w, ", ".join(x for x, _ in kv.most_similar(w, topn=topn))])
        else:
            rows.append([w, "(không có trong từ vựng)"])
    return found, md_table(["từ", f"{topn} láng giềng gần nhất"], rows)


def plot_groups(kv, name):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    words, labels = [], []
    for g, ws in GROUPS.items():
        for w in ws:
            if w in kv.key_to_index and w not in words:
                words.append(w)
                labels.append(g)
    if len(words) < 10:
        return
    X = np.stack([kv[w] for w in words])
    cmap = plt.get_cmap("tab10")
    gnames = list(GROUPS)
    for method in ("pca", "tsne"):
        if method == "pca":
            Y = PCA(n_components=2, random_state=0).fit_transform(X)
        else:
            Y = TSNE(n_components=2, perplexity=min(15, len(words) - 1), init="pca",
                     learning_rate="auto", random_state=0).fit_transform(X)
        fig, ax = plt.subplots(figsize=(8, 6))
        for gi, g in enumerate(gnames):
            idx = [i for i, l in enumerate(labels) if l == g]
            if idx:
                ax.scatter(Y[idx, 0], Y[idx, 1], color=cmap(gi % 10), label=g, s=25)
        for i, w in enumerate(words):
            ax.annotate(w, Y[i], fontsize=6, alpha=0.8)
        ax.legend(fontsize=7)
        ax.set_title(f"{method.upper()} - {name}")
        fig.tight_layout()
        fig.savefig(config.FIG_DIR / f"{method}_{name}.png", dpi=150)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no_viz", action="store_true")
    ap.add_argument("--models", nargs="*", default=None, help="tên mô hình cần đánh giá (mặc định: tất cả *.kv)")
    args = ap.parse_args()

    questions = build_questions()
    apath = config.RESULT_DIR / "analogy_vi.txt"
    write_analogy_file(apath, questions)
    paths = sorted(config.MODEL_DIR.glob("*.kv"))
    for p in paths:
        name = p.stem
        if args.models and name not in args.models:
            continue
        kv = KeyedVectors.load(str(p))
        with Timer() as t:
            res = analogy_eval(kv, questions, apath)
            found, md = neighbors_table(kv, SAMPLE_WORDS)
            (config.TABLE_DIR / f"neighbors_{name}.md").write_text(md, encoding="utf-8")
            if not args.no_viz:
                plot_groups(kv, name)
        res["n_sample_words_found"] = found
        res["vocab_size"] = len(kv.key_to_index)
        if found < 15:
            print(f"[CẢNH BÁO] {name}: chỉ {found} từ mẫu có trong từ vựng (<15). Hãy sửa SAMPLE_WORDS.")
        log_run("intrinsic", name, 0, {"analogy_questions": sum(len(v) for v in questions.values())}, res, t.dt)
        print(f"[{name}] analogy_acc={res['analogy_acc']:.3f} "
              f"covered={res['analogy_covered']}/{res['analogy_total']} sample_found={found}")


if __name__ == "__main__":
    main()
