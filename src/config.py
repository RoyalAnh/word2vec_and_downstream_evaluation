"""Cấu hình chung của dự án (đường dẫn, siêu tham số lưới thí nghiệm)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Trên Kaggle nên đặt: BTL_WORK=/kaggle/working/work
WORK = Path(os.environ.get("BTL_WORK", ROOT / "work"))
DATA_DIR = WORK / "data"
MODEL_DIR = WORK / "models"
RESULT_DIR = WORK / "results"
FIG_DIR = RESULT_DIR / "figures"
TABLE_DIR = RESULT_DIR / "tables"
LOG_CSV = RESULT_DIR / "experiment_log.csv"   # nhật ký thí nghiệm (nộp kèm)
for _d in (DATA_DIR, MODEL_DIR, MODEL_DIR / "ablation", FIG_DIR, TABLE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 43, 44]          # mỗi cấu hình ngẫu nhiên chạy 3 seed
MIN_COUNT = 5                 # tần suất tối thiểu của từ khi huấn luyện embedding

# Lưới word2vec: 2 chế độ x 2 window x 2 số chiều = 8 mô hình
W2V_MODES = ["sg", "cbow"]
W2V_WINDOWS = [3, 8]
W2V_DIMS = [100, 300]
W2V_EPOCHS = 5
W2V_NEGATIVE = 5


def w2v_name(mode, window, dim):
    return f"w2v_{mode}_w{window}_d{dim}"
