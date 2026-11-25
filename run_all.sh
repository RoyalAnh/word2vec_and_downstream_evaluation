#!/usr/bin/env bash
# Chạy toàn bộ quy trình theo đúng thứ tự. Tuỳ chỉnh các tham số ở đầu file nếu cần.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONHASHSEED=0
export BTL_WORK="${BTL_WORK:-$(pwd)/work}"
N_ARTICLES="${N_ARTICLES:-30000}"        # số bài Wikipedia (giảm nếu tách từ quá lâu)
VSFC_DIR="${VSFC_DIR:-}"                  # thư mục UIT-VSFC cục bộ (để trống = tải qua Hugging Face)
NEURAL_KV="${NEURAL_KV:-w2v_sg_w3_d300}" # mô hình dùng khởi tạo cho CNN/LSTM
ABL_CFG="${ABL_CFG:-sg_w3_d300}"         # cấu hình w2v dùng cho ablation ngữ liệu

python src/prepare_data.py --n_articles "$N_ARTICLES" ${VSFC_DIR:+--vsfc_dir "$VSFC_DIR"}
python src/train_w2v.py --workers 1
python src/cooc_svd_glove.py
python src/eval_intrinsic.py
for TASK in sentiment topic; do
  python src/eval_extrinsic.py --task "$TASK" --stage all --neural_kv "$NEURAL_KV"
  python src/eval_phobert.py   --task "$TASK" --mode both
done
python src/corpus_ablation.py --cfg "$ABL_CFG" --task sentiment
python src/bias_analysis.py
python src/aggregate.py --task sentiment
python src/aggregate.py --task topic || true
