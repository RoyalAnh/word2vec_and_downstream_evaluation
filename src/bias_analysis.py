"""Bước 7 (yêu cầu khác) - Khảo sát thiên kiến giới trong embedding tiếng Việt (nghề nghiệp - giới tính).

  (1) Chiếu từ nghề nghiệp lên 'trục giới tính' (trung bình hiệu vector các cặp nam-nữ). Điểm > 0 nghiêng về nam.
  (2) WEAT (Caliskan et al., 2017): effect size + p-value hoán vị, giữa {từ nam, từ nữ} và {nghề/quyền lực, gia đình}.
Danh sách từ chỉ có tác dụng khi từ nằm trong từ vựng; từ thiếu bị bỏ và ghi lại. Đây là phép đo mô tả thiên kiến
trong NGỮ LIỆU - không phải khẳng định về con người; cần thận trọng khi diễn giải (xem báo cáo).
"""
import argparse

import numpy as np
from gensim.models import KeyedVectors

import config
from utils import Timer, log_run, md_table

GENDER_PAIRS = [("anh", "chị"), ("ông", "bà"), ("bố", "mẹ"), ("nam", "nữ"), ("con_trai", "con_gái"),
                ("chồng", "vợ"), ("đàn_ông", "phụ_nữ"), ("trai", "gái")]
OCCUPATIONS = ["bác_sĩ", "y_tá", "giáo_viên", "giảng_viên", "kỹ_sư", "lập_trình_viên", "thư_ký", "giám_đốc",
               "công_nhân", "nông_dân", "luật_sư", "ca_sĩ", "nhà_báo", "thợ_may", "kế_toán", "nhân_viên",
               "lãnh_đạo", "nhà_khoa_học", "đầu_bếp", "tài_xế", "cảnh_sát", "bộ_đội", "diễn_viên", "họa_sĩ",
               "nhạc_sĩ", "doanh_nhân", "nội_trợ", "tiếp_viên", "người_mẫu", "thợ_xây"]
WEAT_X = ["anh", "ông", "bố", "nam", "con_trai", "chồng", "đàn_ông", "cậu", "chú", "trai"]     # từ nam
WEAT_Y = ["chị", "bà", "mẹ", "nữ", "con_gái", "vợ", "phụ_nữ", "dì", "cô", "gái"]               # từ nữ
WEAT_A = ["giám_đốc", "lãnh_đạo", "kỹ_sư", "luật_sư", "nhà_khoa_học", "doanh_nhân", "lập_trình_viên",
          "công_việc", "sự_nghiệp", "kinh_doanh"]                                                # nghề/quyền lực
WEAT_B = ["nội_trợ", "gia_đình", "con_cái", "nhà_cửa", "chăm_sóc", "bếp", "con", "hôn_nhân", "tổ_ấm",
          "việc_nhà"]                                                                            # gia đình


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def gender_projection(kv):
    diffs = [unit(kv[m]) - unit(kv[f]) for m, f in GENDER_PAIRS if m in kv.key_to_index and f in kv.key_to_index]
    if len(diffs) < 3:
        return None, None
    d = unit(np.mean(diffs, axis=0))
    rows = sorted(((w, float(unit(kv[w]) @ d)) for w in OCCUPATIONS if w in kv.key_to_index), key=lambda x: -x[1])
    return len(diffs), rows


def weat(kv, X, Y, A, B, n_perm=10000, seed=0):
    f = lambda ws: [unit(kv[w]) for w in ws if w in kv.key_to_index]
    X, Y, A, B = f(X), f(Y), f(A), f(B)
    if min(len(X), len(Y), len(A), len(B)) < 3:
        return None
    s = lambda w: np.mean([w @ a for a in A]) - np.mean([w @ b for b in B])
    sx, sy = np.array([s(w) for w in X]), np.array([s(w) for w in Y])
    allv = np.concatenate([sx, sy])
    eff = (sx.mean() - sy.mean()) / (allv.std(ddof=1) + 1e-12)
    stat = sx.sum() - sy.sum()
    rng = np.random.RandomState(seed)
    cnt = 0
    for _ in range(n_perm):
        p = rng.permutation(len(allv))
        if allv[p[:len(sx)]].sum() - allv[p[len(sx):]].sum() >= stat:
            cnt += 1
    return {"effect_size": float(eff), "p_value": float((cnt + 1) / (n_perm + 1)),
            "n_X": len(X), "n_Y": len(Y), "n_A": len(A), "n_B": len(B)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    args = ap.parse_args()
    for p in sorted(config.MODEL_DIR.glob("*.kv")):
        name = p.stem
        if args.models and name not in args.models:
            continue
        kv = KeyedVectors.load(str(p))
        with Timer() as t:
            npairs, rows = gender_projection(kv)
            wt = weat(kv, WEAT_X, WEAT_Y, WEAT_A, WEAT_B)
        if rows is None:
            print(f"[{name}] thiếu từ để dựng trục giới tính - bỏ qua")
            continue
        (config.TABLE_DIR / f"bias_projection_{name}.md").write_text(
            md_table(["nghề", "điểm chiếu (>0: nghiêng nam)"], [[w, f"{s:+.3f}"] for w, s in rows]),
            encoding="utf-8")
        met = {"n_gender_pairs": npairs, "n_occupations": len(rows),
               "most_male": [w for w, _ in rows[:5]], "most_female": [w for w, _ in rows[-5:]]}
        if wt:
            met.update(wt)
        log_run("bias_weat", name, 0, {"n_perm": 10000}, met, t.dt)
        print(f"[{name}] WEAT: {wt}")


if __name__ == "__main__":
    main()
