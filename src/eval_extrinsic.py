"""Bước 4 - Đánh giá NGOẠI TẠI trên UIT-VSFC (phân loại cảm xúc `sentiment` hoặc chủ đề `topic`).

Các cách biểu diễn được so sánh (đề bài yêu cầu 4 nhóm):
  1. Baseline rời rạc : BoW nhị phân + LogReg  và  TF-IDF(1-2gram) + LogReg           (stage: linear)
  2. Embedding trung bình + MLP (cho MỌI mô hình *.kv: word2vec / SVD / GloVe)          (stage: mean_mlp)
  3. Embedding + TextCNN / BiLSTM: khởi tạo ngẫu nhiên vs w2v (đóng băng / fine-tune)  (stage: neural)
  4. Embedding tiền huấn luyện PhoBERT + classifier                                    (xem eval_phobert.py)
Chọn siêu tham số bằng tập dev; báo cáo trên tập test; mỗi cấu hình có seed chạy lặp (mean ± std).
"""
import argparse
import collections
import copy
import time

import numpy as np
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

import config
from utils import Timer, log_run, read_jsonl, set_seed

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:  # cho phép chạy stage linear / mean_mlp khi chưa có torch
    torch = None


# ----------------------------------------------------------------- dữ liệu
def load_vsfc_data(task):
    d = {}
    for split in ("train", "dev", "test"):
        rows = read_jsonl(config.DATA_DIR / f"vsfc_{split}.jsonl")
        d[split] = {"tokens": [r["tok"].split() for r in rows],
                    "y": np.array([int(r[task]) for r in rows])}
    d["n_classes"] = int(d["train"]["y"].max()) + 1
    return d


def metrics_of(y_true, y_pred):
    return {"acc": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
            "f1_per_class": [float(x) for x in f1_score(y_true, y_pred, average=None)]}


# ------------------------------------------------- 1. Baseline rời rạc (LR)
def run_linear_baselines(data, task):
    docs = {s: [" ".join(t) for t in data[s]["tokens"]] for s in ("train", "dev", "test")}
    specs = {
        "bow_binary_lr": lambda: CountVectorizer(tokenizer=str.split, token_pattern=None, lowercase=False,
                                                 binary=True, min_df=2),
        "tfidf_lr": lambda: TfidfVectorizer(tokenizer=str.split, token_pattern=None, lowercase=False,
                                            ngram_range=(1, 2), min_df=2, sublinear_tf=True),
    }
    for name, make_vec in specs.items():
        with Timer() as t:
            vec = make_vec()
            Xtr = vec.fit_transform(docs["train"])
            Xdev, Xte = vec.transform(docs["dev"]), vec.transform(docs["test"])
            best = None
            for C in (0.1, 1, 10, 100):                      # chọn C bằng dev macro-F1
                clf = LogisticRegression(C=C, max_iter=3000).fit(Xtr, data["train"]["y"])
                f1 = f1_score(data["dev"]["y"], clf.predict(Xdev), average="macro")
                if best is None or f1 > best[0]:
                    best = (f1, C, clf)
            f1_dev, C, clf = best
            m = metrics_of(data["test"]["y"], clf.predict(Xte))
        met = {"dev_macro_f1": float(f1_dev), "test_acc": m["acc"], "test_macro_f1": m["macro_f1"],
               "test_f1_per_class": m["f1_per_class"], "n_features": int(Xtr.shape[1]), "best_C": C}
        log_run("extrinsic_linear", name, 0, {"task": task}, met, t.dt)
        print(f"[linear] {name}: test acc={m['acc']:.4f} macro-F1={m['macro_f1']:.4f} (C={C})")


# ------------------------------------------- 2. Embedding trung bình + MLP
def mean_vectors(kv, token_lists):
    out = np.zeros((len(token_lists), kv.vector_size), np.float32)
    n_tok = n_hit = 0
    for i, toks in enumerate(token_lists):
        idx = [kv.key_to_index[t] for t in toks if t in kv.key_to_index]
        n_tok += len(toks)
        n_hit += len(idx)
        if idx:
            out[i] = kv.vectors[idx].mean(0)
    return out, (n_hit / max(n_tok, 1))


def eval_mean_mlp(kv, data, name, seeds, task, experiment="extrinsic_mean_mlp", log=True):
    Xtr, _ = mean_vectors(kv, data["train"]["tokens"])
    Xdev, _ = mean_vectors(kv, data["dev"]["tokens"])
    Xte, cov = mean_vectors(kv, data["test"]["tokens"])
    sc = StandardScaler().fit(Xtr)
    Xtr, Xdev, Xte = sc.transform(Xtr), sc.transform(Xdev), sc.transform(Xte)
    results = []
    for seed in seeds:
        set_seed(seed)
        with Timer() as t:
            clf = MLPClassifier(hidden_layer_sizes=(256,), alpha=1e-4, batch_size=64, early_stopping=True,
                                n_iter_no_change=10, max_iter=200, random_state=seed)
            clf.fit(Xtr, data["train"]["y"])
            m_dev = metrics_of(data["dev"]["y"], clf.predict(Xdev))
            m = metrics_of(data["test"]["y"], clf.predict(Xte))
        met = {"dev_macro_f1": m_dev["macro_f1"], "test_acc": m["acc"], "test_macro_f1": m["macro_f1"],
               "test_f1_per_class": m["f1_per_class"], "token_coverage_test": float(cov)}
        results.append(met)
        if log:
            log_run(experiment, name, seed, {"task": task, "hidden": 256}, met, t.dt)
    return results


# --------------------------------------------- 3. Mạng nơ-ron (TextCNN / BiLSTM)
if torch is not None:
    class TextCNN(nn.Module):
        def __init__(self, emb, n_cls, freeze, kernel_sizes=(3, 4, 5), n_filters=100, dropout=0.5):
            super().__init__()
            self.emb = nn.Embedding.from_pretrained(emb, freeze=freeze, padding_idx=0)
            d = emb.shape[1]
            self.convs = nn.ModuleList([nn.Conv1d(d, n_filters, k, padding=k // 2) for k in kernel_sizes])
            self.drop = nn.Dropout(dropout)
            self.fc = nn.Linear(n_filters * len(kernel_sizes), n_cls)

        def forward(self, x, lengths):
            e = self.emb(x).transpose(1, 2)                       # (B, D, L)
            h = [torch.relu(c(e)).max(dim=2).values for c in self.convs]
            return self.fc(self.drop(torch.cat(h, dim=1)))

    class BiLSTM(nn.Module):
        def __init__(self, emb, n_cls, freeze, hidden=128, dropout=0.5):
            super().__init__()
            self.emb = nn.Embedding.from_pretrained(emb, freeze=freeze, padding_idx=0)
            self.lstm = nn.LSTM(emb.shape[1], hidden, batch_first=True, bidirectional=True)
            self.drop = nn.Dropout(dropout)
            self.fc = nn.Linear(2 * hidden, n_cls)

        def forward(self, x, lengths):
            e = self.drop(self.emb(x))
            packed = nn.utils.rnn.pack_padded_sequence(e, lengths.cpu(), batch_first=True, enforce_sorted=False)
            _, (h, _) = self.lstm(packed)
            return self.fc(self.drop(torch.cat([h[-2], h[-1]], dim=1)))


def build_vocab(train_tokens, min_freq=2):
    cnt = collections.Counter(t for toks in train_tokens for t in toks)
    itos = ["<pad>", "<unk>"] + [w for w, c in cnt.most_common() if c >= min_freq]
    return itos, {w: i for i, w in enumerate(itos)}


def encode(token_lists, stoi, max_len):
    X = np.zeros((len(token_lists), max_len), dtype=np.int64)
    L = np.ones(len(token_lists), dtype=np.int64)
    for i, toks in enumerate(token_lists):
        ids = [stoi.get(t, 1) for t in toks[:max_len]]
        X[i, :len(ids)] = ids
        L[i] = max(len(ids), 1)
    return torch.tensor(X), torch.tensor(L)


def build_emb_matrix(itos, kv, use_pretrained, seed):
    rng = np.random.RandomState(seed)
    M = rng.normal(0, 0.1, (len(itos), kv.vector_size)).astype(np.float32)
    M[0] = 0
    hit = 0
    if use_pretrained:
        for i, w in enumerate(itos):
            if w in kv.key_to_index:
                M[i] = kv[w]
                hit += 1
    return torch.tensor(M), hit


@torch.no_grad() if torch is not None else (lambda f: f)
def predict(model, X, L, device, bs=256):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        out.append(model(X[i:i + bs].to(device), L[i:i + bs]).argmax(1).cpu())
    return torch.cat(out).numpy()


def run_neural_once(arch, data, kv, use_pretrained, freeze, seed, device, epochs, batch_size, lr, patience, max_len):
    set_seed(seed)
    itos, stoi = build_vocab(data["train"]["tokens"])
    Xtr, Ltr = encode(data["train"]["tokens"], stoi, max_len)
    Xdev, Ldev = encode(data["dev"]["tokens"], stoi, max_len)
    Xte, Lte = encode(data["test"]["tokens"], stoi, max_len)
    ytr = torch.tensor(data["train"]["y"])
    emb, hit = build_emb_matrix(itos, kv, use_pretrained, seed)
    cls = TextCNN if arch == "textcnn" else BiLSTM
    model = cls(emb, data["n_classes"], freeze).to(device)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    g = torch.Generator().manual_seed(seed)
    dl = DataLoader(TensorDataset(Xtr, Ltr, ytr), batch_size=batch_size, shuffle=True, generator=g)
    lossf = nn.CrossEntropyLoss()
    best = {"dev_f1": -1, "epoch": 0, "state": None}
    bad = 0
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        for xb, lb, yb in dl:
            opt.zero_grad()
            loss = lossf(model(xb.to(device), lb), yb.to(device))
            loss.backward()
            opt.step()
        f1 = f1_score(data["dev"]["y"], predict(model, Xdev, Ldev, device), average="macro")
        if f1 > best["dev_f1"]:
            best = {"dev_f1": f1, "epoch": ep, "state": copy.deepcopy(model.state_dict())}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best["state"])
    m = metrics_of(data["test"]["y"], predict(model, Xte, Lte, device))
    return {"dev_macro_f1": float(best["dev_f1"]), "test_acc": m["acc"], "test_macro_f1": m["macro_f1"],
            "test_f1_per_class": m["f1_per_class"], "best_epoch": best["epoch"],
            "trainable_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
            "pretrained_hit_rate": hit / len(itos) if use_pretrained else 0.0}, time.time() - t0


def run_neural(data, task, kv_name, seeds, epochs, batch_size, lr, patience, max_len):
    if torch is None:
        raise SystemExit("Cần cài torch để chạy stage neural.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    kv = KeyedVectors.load(str(config.MODEL_DIR / f"{kv_name}.kv"))
    settings = [("random", False, False), ("w2v", True, True), ("w2v", True, False)]  # (init, pretrained, freeze)
    for arch in ("textcnn", "bilstm"):
        for init, pre, freeze in settings:
            name = f"{arch}|{init}|{'frozen' if freeze else 'finetune'}|{kv_name}"
            for seed in seeds:
                met, dt = run_neural_once(arch, data, kv, pre, freeze, seed, device, epochs, batch_size,
                                          lr, patience, max_len)
                cfg = {"task": task, "arch": arch, "init": init, "freeze": freeze, "kv": kv_name,
                       "epochs": epochs, "batch_size": batch_size, "lr": lr, "max_len": max_len}
                log_run("extrinsic_neural", name, seed, cfg, met, dt)
                print(f"[neural] {name} seed={seed}: test macro-F1={met['test_macro_f1']:.4f} "
                      f"(best epoch {met['best_epoch']}, {dt:.0f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="sentiment", choices=["sentiment", "topic"])
    ap.add_argument("--stage", default="all", choices=["all", "linear", "mean_mlp", "neural"])
    ap.add_argument("--neural_kv", default="w2v_sg_w3_d300", help="mô hình dùng khởi tạo cho TextCNN/BiLSTM")
    ap.add_argument("--seeds", type=int, nargs="*", default=config.SEEDS)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max_len", type=int, default=64)
    args = ap.parse_args()

    data = load_vsfc_data(args.task)
    if args.stage in ("all", "linear"):
        run_linear_baselines(data, args.task)
    if args.stage in ("all", "mean_mlp"):
        for p in sorted(config.MODEL_DIR.glob("*.kv")):
            kv = KeyedVectors.load(str(p))
            res = eval_mean_mlp(kv, data, p.stem, args.seeds, args.task)
            f1s = [r["test_macro_f1"] for r in res]
            print(f"[mean_mlp] {p.stem}: macro-F1 = {np.mean(f1s):.4f} ± {np.std(f1s, ddof=1) if len(f1s) > 1 else 0:.4f}")
    if args.stage in ("all", "neural"):
        run_neural(data, args.task, args.neural_kv, args.seeds, args.epochs, args.batch_size, args.lr,
                   args.patience, args.max_len)


if __name__ == "__main__":
    main()
