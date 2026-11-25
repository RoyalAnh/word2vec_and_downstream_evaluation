"""Bước 5 - Embedding tiền huấn luyện PhoBERT (vinai/phobert-base) + classifier, trên UIT-VSFC.

  --mode frozen   : trích đặc trưng (mean-pooling lớp cuối, backbone đóng băng) + LogReg
  --mode finetune : fine-tune toàn mạng (AdamW, warmup tuyến tính, chọn epoch theo dev macro-F1), 3 seed
PhoBERT yêu cầu văn bản đã TÁCH TỪ (ở đây dùng cùng bộ tách từ với phần còn lại, trường `seg` của jsonl).
"""
import argparse
import time

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

import config
from utils import Timer, log_run, read_jsonl, set_seed

MODEL_ID = "vinai/phobert-base"


def load_split(split, task):
    rows = read_jsonl(config.DATA_DIR / f"vsfc_{split}.jsonl")
    return [r["seg"] for r in rows], np.array([int(r[task]) for r in rows])


def mets(y, p):
    return {"acc": float(accuracy_score(y, p)), "macro_f1": float(f1_score(y, p, average="macro")),
            "f1_per_class": [float(x) for x in f1_score(y, p, average=None)]}


@torch.no_grad()
def extract(model, tok, texts, device, max_len, bs=64):
    model.eval()
    feats = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
        h = model(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).float()
        feats.append(((h * m).sum(1) / m.sum(1)).cpu().numpy())
    return np.concatenate(feats)


def run_frozen(task, device, max_len):
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
    model = AutoModel.from_pretrained(MODEL_ID).to(device)
    data = {s: load_split(s, task) for s in ("train", "dev", "test")}
    with Timer() as t:
        X = {s: extract(model, tok, data[s][0], device, max_len) for s in data}
        sc = StandardScaler().fit(X["train"])
        best = None
        for C in (0.01, 0.1, 1, 10):
            clf = LogisticRegression(C=C, max_iter=3000).fit(sc.transform(X["train"]), data["train"][1])
            f1 = f1_score(data["dev"][1], clf.predict(sc.transform(X["dev"])), average="macro")
            if best is None or f1 > best[0]:
                best = (f1, C, clf)
        f1_dev, C, clf = best
        m = mets(data["test"][1], clf.predict(sc.transform(X["test"])))
    met = {"dev_macro_f1": float(f1_dev), "test_acc": m["acc"], "test_macro_f1": m["macro_f1"],
           "test_f1_per_class": m["f1_per_class"], "best_C": C}
    log_run("extrinsic_phobert", "phobert_frozen_meanpool_lr", 0, {"task": task, "model": MODEL_ID}, met, t.dt)
    print(f"[phobert frozen] test acc={m['acc']:.4f} macro-F1={m['macro_f1']:.4f}")


@torch.no_grad()
def predict(model, tok, texts, device, max_len, bs=64):
    model.eval()
    out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
        out.append(model(**enc).logits.argmax(1).cpu())
    return torch.cat(out).numpy()


def run_finetune(task, device, seeds, epochs, lr, bs, max_len):
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
    data = {s: load_split(s, task) for s in ("train", "dev", "test")}
    n_cls = int(data["train"][1].max()) + 1
    for seed in seeds:
        set_seed(seed)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, num_labels=n_cls).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        n = len(data["train"][0])
        steps = epochs * ((n + bs - 1) // bs)
        sch = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
        use_amp = device == "cuda"
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        rng = np.random.RandomState(seed)
        best = {"dev_f1": -1, "epoch": 0, "test": None}
        t0 = time.time()
        for ep in range(1, epochs + 1):
            model.train()
            perm = rng.permutation(n)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                enc = tok([data["train"][0][j] for j in idx], padding=True, truncation=True,
                          max_length=max_len, return_tensors="pt").to(device)
                y = torch.tensor(data["train"][1][idx]).to(device)
                opt.zero_grad()
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                    loss = model(**enc, labels=y).loss
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sch.step()
            f1 = f1_score(data["dev"][1], predict(model, tok, data["dev"][0], device, max_len), average="macro")
            if f1 > best["dev_f1"]:            # chỉ đo test khi dev tốt lên (không cần lưu trọng số)
                best = {"dev_f1": f1, "epoch": ep,
                        "test": mets(data["test"][1], predict(model, tok, data["test"][0], device, max_len))}
        dt = time.time() - t0
        m = best["test"]
        met = {"dev_macro_f1": float(best["dev_f1"]), "test_acc": m["acc"], "test_macro_f1": m["macro_f1"],
               "test_f1_per_class": m["f1_per_class"], "best_epoch": best["epoch"]}
        cfg = {"task": task, "model": MODEL_ID, "epochs": epochs, "lr": lr, "batch_size": bs, "max_len": max_len}
        log_run("extrinsic_phobert", "phobert_finetune", seed, cfg, met, dt)
        print(f"[phobert finetune] seed={seed}: test acc={m['acc']:.4f} macro-F1={m['macro_f1']:.4f} "
              f"(best epoch {best['epoch']}, {dt:.0f}s)")
        del model, opt
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="sentiment", choices=["sentiment", "topic"])
    ap.add_argument("--mode", default="both", choices=["frozen", "finetune", "both"])
    ap.add_argument("--seeds", type=int, nargs="*", default=config.SEEDS)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=128)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.mode in ("frozen", "both"):
        run_frozen(args.task, device, args.max_len)
    if args.mode in ("finetune", "both"):
        run_finetune(args.task, device, args.seeds, args.epochs, args.lr, args.batch_size, args.max_len)


if __name__ == "__main__":
    main()
