# BTL Học sâu – Đề tài 8: Biểu diễn từ tiếng Việt (word2vec) và đánh giá qua bài toán downstream

**Hình thức thực hiện:** Cá nhân (01 sinh viên).

Toàn bộ quy trình: ngữ liệu Wikipedia tiếng Việt → tiền xử lý & tách từ → word2vec (Skip-gram/CBOW) → đánh giá nội tại
(láng giềng, analogy, t-SNE/PCA) → đánh giá ngoại tại trên UIT-VSFC với 4 cách biểu diễn → phần "yêu cầu khác"
(SVD/GloVe, thiên kiến giới) → ablation ngữ liệu → tổng hợp bảng/biểu đồ từ nhật ký thí nghiệm.

## 1. Cài đặt

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
Python ≥ 3.10. Cần GPU (Kaggle T4 là đủ) cho: TextCNN/BiLSTM, GloVe, fine-tune PhoBERT. Các phần còn lại chạy được trên CPU.

## 2. Chạy lại toàn bộ

```bash
bash run_all.sh                      # chạy tất cả theo thứ tự (biến môi trường tuỳ chỉnh ở đầu file)
```
Hoặc từng bước (mọi đầu ra nằm trong `work/`, đổi bằng biến môi trường `BTL_WORK`):

| # | Lệnh | Việc làm | Đầu ra chính |
|---|------|----------|--------------|
| 1 | `python src/prepare_data.py --n_articles 30000` | tải/chuẩn hoá/tách từ, thống kê, Zipf | `data/*.txt|jsonl`, `results/data_stats.json`, `figures/zipf_*.png` |
| 2 | `python src/train_w2v.py --workers 1` | 8 mô hình word2vec (sg/cbow × window 3,8 × dim 100,300) | `models/w2v_*.kv` |
| 3 | `python src/cooc_svd_glove.py` | PPMI+SVD và GloVe cùng ngữ liệu | `models/svd_*.kv`, `models/glove_*.kv` |
| 4 | `python src/eval_intrinsic.py` | láng giềng (≥15 từ), analogy, PCA/t-SNE | `tables/neighbors_*.md`, `figures/tsne_*.png` |
| 5 | `python src/eval_extrinsic.py --task sentiment --stage all` | BoW/TF-IDF+LR, embedding TB+MLP, TextCNN/BiLSTM | dòng log trong `experiment_log.csv` |
| 6 | `python src/eval_phobert.py --task sentiment --mode both` | PhoBERT đóng băng + LR; fine-tune | dòng log |
| 7 | `python src/corpus_ablation.py --cfg sg_w3_d300` | ablation kích thước/miền ngữ liệu | `figures/corpus_ablation.png` |
| 8 | `python src/bias_analysis.py` | thiên kiến giới (chiếu trục + WEAT) | `tables/bias_*.md` |
| 9 | `python src/aggregate.py --task sentiment` | gộp bảng + biểu đồ từ log | `results/summary_sentiment.md`, `tables/table*.md` |

Lặp lại các bước 5, 6, 9 với `--task topic` để có thêm bài toán phân loại chủ đề.
Kiểm tra nhanh đường ống bằng dữ liệu **giả** (vài giây, không tải gì): `python src/prepare_data.py --smoke --n_jobs 1`
(dùng `BTL_WORK=/tmp/smoke` để không lẫn với kết quả thật).

### Chạy trên Kaggle
1. Nén thư mục dự án, upload thành Kaggle Dataset; tạo notebook, Add Data, bật **Internet** và **GPU T4**.
2. Mở `notebooks/kaggle_run.ipynb` (hoặc dán các ô lệnh). Kết quả ở `/kaggle/working/work/` → tải `results.zip` về.
3. Nếu `load_dataset("uitnlp/vietnamese_students_feedback")` báo lỗi (phiên bản `datasets` mới không hỗ trợ dataset dạng script),
   tải UIT-VSFC từ nguồn gốc (UIT NLP Group) hoặc bản mirror trên Kaggle rồi chạy `--vsfc_dir <thư_mục>`;
   hỗ trợ cấu trúc `train|dev|test/{sentences,sentiments,topics}.txt` hoặc `train.csv/dev.csv/test.csv` (cột `sentence,sentiment,topic`).
   Tương tự có thể dùng `--wiki_txt` nếu không tải được Wikipedia.

## 3. Tái lập (reproducibility)
- Seed cố định (`utils.set_seed`); word2vec: `workers=1` + hàm băm ổn định (`utils.stable_hash`) ⇒ cùng mã, cùng seed, cùng số.
  Nếu dùng `--workers >1` thì kết quả có thể lệch rất nhỏ – phải ghi rõ trong báo cáo.
- Mô hình thần kinh chạy 3 seed (42, 43, 44); báo cáo trung bình ± độ lệch chuẩn mẫu. GPU có thể gây sai khác nhỏ ở mức 1e-3.
- Chọn siêu tham số/epoch bằng tập dev, báo cáo trên tập test (không chọn theo test).
- **Mọi lượt chạy** được ghi vào `work/results/experiment_log.csv` (thời điểm, thí nghiệm, tên cấu hình, seed, cấu hình JSON,
  kết quả JSON, thời gian). Bảng trong báo cáo phải được lấy từ `aggregate.py`, không gõ tay.

## 4. Cấu trúc thư mục
```
src/            mã nguồn (config, utils, prepare_data, train_w2v, cooc_svd_glove, eval_*, bias_analysis, corpus_ablation, aggregate)
notebooks/      colab_run.ipynb, kaggle_run.ipynb
run_all.sh      chạy tất cả
work/           (sinh ra khi chạy) data, models, results
```

## 5. Nguồn tham khảo và phần thực hiện
- Thư viện: gensim (word2vec), pyvi (tách từ), scikit-learn, PyTorch, Hugging Face `datasets`/`transformers`.
- Dữ liệu: Wikipedia tiếng Việt (`wikimedia/wikipedia`, bản 20231101.vi, giấy phép CC BY-SA); UIT-VSFC (Nguyen et al., KSE 2018).
- Mô hình tiền huấn luyện: PhoBERT (`vinai/phobert-base`, Nguyen & Nguyen, 2020).
- Các script trong `src/` được phát triển cho bài tập này và sử dụng thư viện nêu trên.

## 6. Hạn chế đã biết
- Danh sách từ analogy / nhóm ngữ nghĩa / từ nghề nghiệp (đầu `eval_intrinsic.py`, `bias_analysis.py`) viết tay theo cách tách từ
  của pyvi; từ không có trong từ vựng sẽ bị bỏ và được báo ở cột "phủ" – cần rà soát sau lần chạy đầu tiên.
- Ngữ liệu mặc định 30.000 bài Wikipedia (mẫu ngẫu nhiên có seed) là nhỏ so với embedding công bố; đây là một phần của câu hỏi nghiên cứu.
