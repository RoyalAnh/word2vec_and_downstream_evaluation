"""Bước 6 (yêu cầu khác) - Ma trận đồng xuất hiện, PPMI+SVD và GloVe trên CÙNG ngữ liệu với word2vec.

Kết quả được lưu thành *.kv (cùng định dạng với word2vec) nên các script eval_* tự động nhận diện:
  svd_w{window}_d{dim}.kv   và   glove_w{window}_d{dim}.kv
Cửa sổ đồng xuất hiện có trọng số 1/khoảng_cách (như GloVe gốc); từ ngoài từ vựng bị loại trước khi tính cửa sổ
(giống gensim). GloVe cài bằng PyTorch (hàm mất mát có trọng số f(x) = min((x/x_max)^alpha, 1), Adagrad).
"""
import argparse
import collections
from array import array

import numpy as np
import scipy.sparse as sp
from gensim.models import KeyedVectors
from scipy.sparse.linalg import svds

import config
from utils import Timer, log_run, set_seed


def build_vocab(path, min_count, max_vocab):
    cnt = collections.Counter()
    with open(path, encoding="utf-8") as f:
        for line in f:
            cnt.update(line.split())
    items = sorted(((w, c) for w, c in cnt.items() if c >= min_count), key=lambda x: (-x[1], x[0]))[:max_vocab]
    itos = [w for w, _ in items]
    return itos, {w: i for i, w in enumerate(itos)}


def cooccurrence(path, stoi, window):
    ids = array("i")
    with open(path, encoding="utf-8") as f:
        for line in f:
            for w in line.split():
                i = stoi.get(w)
                if i is not None:
                    ids.append(i)
            ids.extend([-1] * window)          # đệm để không có cặp nào vắt ngang hai câu
    ids = np.frombuffer(ids, dtype=np.int32)
    V = len(stoi)
    X = sp.csr_matrix((V, V), dtype=np.float32)
    for d in range(1, window + 1):
        a, b = ids[:-d], ids[d:]
        m = (a >= 0) & (b >= 0)
        a, b = a[m], b[m]
        w = np.full(a.shape[0] * 2, 1.0 / d, dtype=np.float32)
        part = sp.coo_matrix((w, (np.concatenate([a, b]), np.concatenate([b, a]))), shape=(V, V)).tocsr()
        X = X + part
    return X


def ppmi(X, alpha=0.75):
    X = X.tocoo()
    data = X.data.astype(np.float64)
    total = data.sum()
    row = np.asarray(X.sum(1), dtype=np.float64).ravel()
    col = np.asarray(X.sum(0), dtype=np.float64).ravel() ** alpha     # làm mịn phân phối ngữ cảnh
    col /= col.sum()
    pmi = np.log((data / total) / ((row[X.row] / total) * col[X.col]))
    keep = pmi > 0
    return sp.csr_matrix((pmi[keep], (X.row[keep], X.col[keep])), shape=X.shape)


def svd_embeddings(P, dim, seed):
    k = min(dim, min(P.shape) - 1)
    v0 = np.random.RandomState(seed).rand(min(P.shape))
    U, S, _ = svds(P, k=k, v0=v0)
    order = np.argsort(-S)
    return U[:, order] * np.sqrt(S[order])


def glove_embeddings(X, dim, epochs, lr, seed, device, x_max=100.0, alpha=0.75, batch=100_000):
    import torch
    torch.manual_seed(seed)
    coo = X.tocoo()
    I = torch.tensor(coo.row, dtype=torch.long, device=device)
    J = torch.tensor(coo.col, dtype=torch.long, device=device)
    Xv = torch.tensor(coo.data, dtype=torch.float32, device=device)
    logx = torch.log(Xv)
    wt = torch.clamp((Xv / x_max) ** alpha, max=1.0)
    V = X.shape[0]
    W, Wc = torch.nn.Embedding(V, dim).to(device), torch.nn.Embedding(V, dim).to(device)
    b, bc = torch.nn.Embedding(V, 1).to(device), torch.nn.Embedding(V, 1).to(device)
    for emb in (W, Wc):
        torch.nn.init.uniform_(emb.weight, -0.5 / dim, 0.5 / dim)
    for emb in (b, bc):
        torch.nn.init.zeros_(emb.weight)
    opt = torch.optim.Adagrad(list(W.parameters()) + list(Wc.parameters()) + list(b.parameters())
                              + list(bc.parameters()), lr=lr)
    g = torch.Generator(device=device).manual_seed(seed)
    nnz = Xv.shape[0]
    last = float("nan")
    for ep in range(epochs):
        perm = torch.randperm(nnz, device=device, generator=g)
        tot = 0.0
        for s in range(0, nnz, batch):
            idx = perm[s:s + batch]
            i, j = I[idx], J[idx]
            diff = (W(i) * Wc(j)).sum(1) + b(i).squeeze(1) + bc(j).squeeze(1) - logx[idx]
            loss = (wt[idx] * diff ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * idx.shape[0]
        last = tot / nnz
        print(f"   glove epoch {ep + 1}/{epochs} loss={last:.5f}")
    return (W.weight + Wc.weight).detach().cpu().numpy(), last


def save_kv(itos, arr, path):
    kv = KeyedVectors(vector_size=arr.shape[1])
    kv.add_vectors(itos, arr.astype(np.float32))
    kv.save(str(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, nargs="*", default=config.W2V_WINDOWS)
    ap.add_argument("--dims", type=int, nargs="*", default=config.W2V_DIMS)
    ap.add_argument("--max_vocab", type=int, default=30000)
    ap.add_argument("--glove_epochs", type=int, default=15)
    ap.add_argument("--glove_lr", type=float, default=0.05)
    ap.add_argument("--skip_glove", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    corpus = config.DATA_DIR / "wiki_tokens.txt"
    itos, stoi = build_vocab(corpus, config.MIN_COUNT, args.max_vocab)
    print(f"vocab = {len(itos):,}")
    for window in args.windows:
        with Timer() as tc:
            X = cooccurrence(corpus, stoi, window)
        print(f"[cooc] window={window}: nnz={X.nnz:,} ({tc.dt:.0f}s)")
        P = ppmi(X)
        for dim in args.dims:
            set_seed(args.seed)
            name = f"svd_w{window}_d{dim}"
            with Timer() as t:
                emb = svd_embeddings(P, dim, args.seed)
            save_kv(itos, emb, config.MODEL_DIR / f"{name}.kv")
            log_run("train_svd", name, args.seed, {"window": window, "dim": dim, "vocab": len(itos),
                                                   "weighting": "1/d", "ppmi_alpha": 0.75},
                    {"nnz": int(X.nnz), "vocab_size": len(itos)}, t.dt + tc.dt)
            print(f"[svd] {name} ({t.dt:.0f}s)")
        if args.skip_glove:
            continue
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        for dim in args.dims:
            set_seed(args.seed)
            name = f"glove_w{window}_d{dim}"
            with Timer() as t:
                emb, last = glove_embeddings(X, dim, args.glove_epochs, args.glove_lr, args.seed, device)
            save_kv(itos, emb, config.MODEL_DIR / f"{name}.kv")
            log_run("train_glove", name, args.seed, {"window": window, "dim": dim, "epochs": args.glove_epochs,
                                                     "lr": args.glove_lr, "x_max": 100, "alpha": 0.75},
                    {"final_loss": last, "nnz": int(X.nnz), "vocab_size": len(itos)}, t.dt + tc.dt)
            print(f"[glove] {name} ({t.dt:.0f}s)")


if __name__ == "__main__":
    main()
