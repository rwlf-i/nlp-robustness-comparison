
import os
from datetime import datetime


os.makedirs('/content', exist_ok=True)
project_dir = "/content"
run_dir = os.path.join(project_dir, "results")
os.makedirs(run_dir, exist_ok=True)
run_id = datetime.now().strftime("%Y-%m-%d_%H-%M")

print("Папка проекта:", project_dir)
print("Для экспериментов:", run_dir)

# Проверка

import os, time, psutil, random
import numpy as np
import pandas as pd
import sys
sys.path.append("/content/checklist")

import pattern.text.en as en
sys.modules["pattern.en"] = en

from checklist.perturb import Perturb
import augly.text as textaugs
import textattack

# Базовая проверка
print("CheckList typo:", Perturb.add_typos("This movie was terrible."))
print("AugLy zwsp:", textaugs.insert_zero_width_chars("This movie was terrible.")[0])
print("TextAttack module:", textattack)

# Конфиругация

eval_size = 400
train_size = 20000
seeds = [42, 43, 44, 45, 46]

# 3 датасета: SST-2, IMDb, Emotion
# 3 HF-модели: fine-tuned под каждый датасет
DATASET_SPECS = [
    {
        "dataset_key": "sst2",
        "dataset_hf_id": "glue/sst2",
        "load_args": ("glue", "sst2"),
        "train_split": "train",
        "eval_split": "validation",
        "text_col": "sentence",
        "label_col": "label",
        "hf_model_name": "distilbert-base-uncased-finetuned-sst-2-english",
    },
    {
        "dataset_key": "imdb",
        "dataset_hf_id": "imdb",
        "load_args": ("imdb",),
        "train_split": "train",
        "eval_split": "test",
        "text_col": "text",
        "label_col": "label",
        "hf_model_name": "lvwerra/distilbert-imdb",
    },
    {
        "dataset_key": "emotion",
        "dataset_hf_id": "dair-ai/emotion",
        "load_args": ("dair-ai/emotion",),
        "train_split": "train",
        "eval_split": "validation",
        "text_col": "text",
        "label_col": "label",
        "hf_model_name": "transformersbook/distilbert-base-uncased-finetuned-emotion",
    },
]

print("Datasets:", [d["dataset_hf_id"] for d in DATASET_SPECS])
print("HF models:", [d["hf_model_name"] for d in DATASET_SPECS])


import random
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import torch
from sklearn.metrics import accuracy_score, f1_score

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

def compute_acc_f1(y_true, y_pred):
    acc = float(accuracy_score(y_true, y_pred))
    uniq = set(int(x) for x in y_true)
    if len(uniq) <= 2:
        f1 = float(f1_score(y_true, y_pred))
        f1_avg = "binary"
    else:
        f1 = float(f1_score(y_true, y_pred, average="macro"))
        f1_avg = "macro"
    return acc, f1, f1_avg

import csv

def append_row_to_csv(csv_path: str, row: dict, columns: list):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if not file_exists:
            writer.writeheader()
        safe_row = {c: row.get(c, "") for c in columns}
        writer.writerow(safe_row)


COLUMNS = [
    "run_id", "seed", "repeat_id",
    "dataset_key", "dataset_hf_id", "hf_model_name",
    "n_labels", "f1_avg",
    "dataset", "eval_size",
    "model", "tool", "test_id",
    "n_eval", "n_ok", "fail_rate", "changed_rate", "avg_similarity",
    "invariance_rate",
    "acc", "f1",
    "acc_clean", "f1_clean", "drop_acc", "drop_f1",
    "time_sec", "peak_rss_mb",
    "attack_success_rate",
    "notes",
]

results_csv = os.path.join(run_dir, "results.csv")
print("Файл результатов:", results_csv)

from difflib import SequenceMatcher

def calc_similarity_stats(orig_texts, new_texts):
    n_eval = len(orig_texts)

    ok_pairs = []
    for o, n in zip(orig_texts, new_texts):
        if isinstance(n, str):
            ok_pairs.append((o, n))

    n_ok = len(ok_pairs)
    fail_rate = 1.0 - (n_ok / n_eval) if n_eval > 0 else 1.0

    if n_ok == 0:
        return {
            "n_eval": n_eval,
            "n_ok": 0,
            "fail_rate": 1.0,
            "changed_rate": 0.0,
            "avg_similarity": 0.0,
        }

    changed = 0
    sim_sum = 0.0

    for o, n in ok_pairs:
        if n != o:
            changed += 1
        sim_sum += SequenceMatcher(None, o, n).ratio()

    changed_rate = changed / n_ok
    avg_similarity = sim_sum / n_ok

    return {
        "n_eval": n_eval,
        "n_ok": n_ok,
        "fail_rate": fail_rate,
        "changed_rate": changed_rate,
        "avg_similarity": avg_similarity,
    }

from datasets import load_dataset

def load_splits_for_spec(spec: dict):
    ds = load_dataset(*spec["load_args"])

    train_split = spec["train_split"]
    eval_split = spec["eval_split"]

    if train_split not in ds:
        train_split = list(ds.keys())[0]

    if eval_split not in ds:
        # fallback: test, иначе train
        eval_split = "test" if "test" in ds else train_split

    return ds[train_split], ds[eval_split], train_split, eval_split

def sample_texts_labels(split, text_col: str, label_col: str, n: int, seed: int):
    idx = list(range(len(split)))
    set_seed(seed)
    random.shuffle(idx)
    idx = idx[:min(n, len(idx))]

    texts = [split[i][text_col] for i in idx]
    labels = [int(split[i][label_col]) for i in idx]
    return texts, labels

def infer_n_labels(train_split, label_col: str, max_scan: int = 5000):
    # быстро оценим число классов
    k = min(len(train_split), max_scan)
    labs = [int(train_split[i][label_col]) for i in range(k)]
    return len(set(labs))

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from transformers import AutoTokenizer, AutoModelForSequenceClassification

def train_sklearn_model(train_texts, train_labels):
    sk_model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=50000)),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    t0 = time.time()
    sk_model.fit(train_texts, train_labels)
    train_time = time.time() - t0
    return sk_model, float(train_time)

def load_hf_model(hf_model_name: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    hf_model = AutoModelForSequenceClassification.from_pretrained(hf_model_name)
    hf_model.to(device)
    hf_model.eval()
    return hf_model, tokenizer, device

proc = psutil.Process(os.getpid())

def _rss_mb() -> float:
    return proc.memory_info().rss / (1024 * 1024)

def eval_sklearn(model, texts, labels):
    rss0 = _rss_mb()
    peak = rss0
    t0 = time.time()

    preds = model.predict(texts)

    t_sec = time.time() - t0
    peak = max(peak, _rss_mb())
    peak_delta = max(0.0, peak - rss0)

    acc, f1, f1_avg = compute_acc_f1(labels, preds)
    return {"acc": acc, "f1": f1, "f1_avg": f1_avg, "time_sec": float(t_sec), "peak_rss_mb": float(peak_delta), "preds": preds}

def hf_predict_with_peak(hf_model, tokenizer, device, texts, batch_size=32, max_length=256, peak_mb_start=None):
    preds_all = []
    peak = _rss_mb() if peak_mb_start is None else peak_mb_start

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            enc = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = hf_model(**enc)
            preds = torch.argmax(out.logits, dim=-1).detach().cpu().tolist()
            preds_all.extend(preds)

            cur = _rss_mb()
            if cur > peak:
                peak = cur

    return preds_all, peak

def eval_hf(hf_model, tokenizer, texts, labels, device, batch_size=32, max_length=256):
    rss0 = _rss_mb()
    peak = rss0
    t0 = time.time()

    preds, peak = hf_predict_with_peak(hf_model, tokenizer, device, texts, batch_size=batch_size, max_length=max_length, peak_mb_start=peak)

    t_sec = time.time() - t0
    peak_delta = max(0.0, peak - rss0)

    acc, f1, f1_avg = compute_acc_f1(labels, preds)
    return {"acc": acc, "f1": f1, "f1_avg": f1_avg, "time_sec": float(t_sec), "peak_rss_mb": float(peak_delta), "preds": preds}

def _safe_augly_apply(func, t: str):
    try:
        out = func(t)
        if isinstance(out, (list, tuple)):
            if len(out) == 0:
                return None
            out = out[0]
        return out if isinstance(out, str) else None
    except Exception:
        return None

aug_tests = {
    "augly_insert_punct": lambda t: _safe_augly_apply(textaugs.insert_punctuation_chars, t),
    "augly_insert_zwsp":  lambda t: _safe_augly_apply(textaugs.insert_zero_width_chars, t),
    "augly_unicode":      lambda t: _safe_augly_apply(textaugs.replace_similar_unicode_chars, t),
}
if hasattr(textaugs, "insert_whitespace_chars"):
    aug_tests["augly_insert_whitespace"] = lambda t: _safe_augly_apply(textaugs.insert_whitespace_chars, t)

print("AugLy tests:", list(aug_tests.keys()))

def _safe_checklist_apply(func, t: str):
    try:
        out = func(t)
        if isinstance(out, (list, tuple)):
            if len(out) == 0:
                return None
            out = out[0]
        return out if isinstance(out, str) else None
    except Exception:
        return None

cl_tests = {}
if hasattr(Perturb, "add_typos"):
    cl_tests["checklist_typos"] = lambda t: _safe_checklist_apply(Perturb.add_typos, t)

print("CheckList tests:", list(cl_tests.keys()))

import nltk

nltk.download("averaged_perceptron_tagger")
nltk.download("averaged_perceptron_tagger_eng")
nltk.download("punkt")

from textattack.attack_recipes import TextFoolerJin2019
from textattack.datasets import Dataset
from textattack import Attacker, AttackArgs
from textattack.attack_results import SuccessfulAttackResult, FailedAttackResult, SkippedAttackResult
from textattack.models.wrappers import HuggingFaceModelWrapper, SklearnModelWrapper

# Переменные для настройки атаки
N_ATTACK_SK = 40
N_ATTACK_HF = 10
QUERY_BUDGET = 200

BATCH_SIZE = 32
MAX_LENGTH = 256

test_id_textattack = "textattack_textfooler"

class TA_VectorizerAdapter:
    def __init__(self, vectorizer):
        self.v = vectorizer
    def transform(self, texts):
        return self.v.transform(texts)
    def get_feature_names(self):
        if hasattr(self.v, "get_feature_names_out"):
            return list(self.v.get_feature_names_out())
        return list(self.v.get_feature_names())

def run_textattack_for_wrapper(attack, texts, labels, n_attack, query_budget):
    texts_n = texts[:n_attack]
    labels_n = labels[:n_attack]

    ds_ta = Dataset(list(zip(texts_n, labels_n)))
    attack_args = AttackArgs(
        num_examples=n_attack,
        query_budget=query_budget,
        silent=True,
        disable_stdout=True,
        enable_advance_metrics=False
    )

    attacker = Attacker(attack, ds_ta, attack_args)
    results = attacker.attack_dataset()

    n_success = 0
    n_failed = 0
    n_skipped = 0

    orig_attempted = []
    final_attempted = []
    y_attempted = []

    for i, r in enumerate(results):
        orig = texts_n[i]
        y = int(labels_n[i])

        if isinstance(r, SkippedAttackResult):
            n_skipped += 1
            continue

        orig_attempted.append(orig)
        y_attempted.append(y)

        if isinstance(r, SuccessfulAttackResult):
            n_success += 1
            final_attempted.append(r.perturbed_text())
        elif isinstance(r, FailedAttackResult):
            n_failed += 1
            final_attempted.append(orig)
        else:
            n_failed += 1
            final_attempted.append(orig)

    return final_attempted, orig_attempted, y_attempted, n_skipped, n_failed, n_success

def make_stats_for_textattack(orig_attempted, final_attempted, n_attack, n_skipped):
    base = calc_similarity_stats(orig_attempted, final_attempted)
    return {
        "n_eval": n_attack,
        "n_ok": base["n_ok"],
        "fail_rate": (n_skipped / n_attack) if n_attack > 0 else 1.0,
        "changed_rate": base["changed_rate"],
        "avg_similarity": base["avg_similarity"],
    }

def run_one_dataset(spec: dict):
    dataset_key = spec["dataset_key"]
    dataset_hf_id = spec["dataset_hf_id"]
    hf_model_name = spec["hf_model_name"]
    dataset_str = dataset_hf_id

    print("\n" + "="*90)
    print(f"DATASET: {dataset_hf_id} | HF: {hf_model_name}")
    print("="*90)

    train_data, eval_data, train_split_name, eval_split_name = load_splits_for_spec(spec)
    n_labels = infer_n_labels(train_data, spec["label_col"])
    print(f"splits: train={train_split_name} ({len(train_data)}), eval={eval_split_name} ({len(eval_data)})")
    print("n_labels:", n_labels)

    # подготовка набора для SK
    seed0 = seeds[0]
    train_texts, train_labels = sample_texts_labels(train_data, spec["text_col"], spec["label_col"], train_size, seed0)

    # train SK
    sk_model, train_time = train_sklearn_model(train_texts, train_labels)
    print("SK train time (sec):", round(train_time, 2))

    # загрузка HF
    hf_model, tokenizer, device = load_hf_model(hf_model_name)
    print("Device:", device)

    # clean запуск
    clean_baselines = {}

    for repeat_id, seed in enumerate(seeds, start=1):
        set_seed(seed)

        texts, labels = sample_texts_labels(eval_data, spec["text_col"], spec["label_col"], eval_size, seed)

        sk_clean = eval_sklearn(sk_model, texts, labels)
        hf_clean = eval_hf(hf_model, tokenizer, texts, labels, device=device, batch_size=32, max_length=256)

        clean_baselines[repeat_id] = {
            "seed": seed,
            "texts": texts,
            "labels": labels,
            "sk": {"acc_clean": sk_clean["acc"], "f1_clean": sk_clean["f1"], "preds_clean": sk_clean["preds"], "f1_avg": sk_clean["f1_avg"]},
            "hf": {"acc_clean": hf_clean["acc"], "f1_clean": hf_clean["f1"], "preds_clean": hf_clean["preds"], "f1_avg": hf_clean["f1_avg"]},
        }

        row_sk = {
            "run_id": run_id, "seed": seed, "repeat_id": repeat_id,
            "dataset_key": dataset_key, "dataset_hf_id": dataset_hf_id, "hf_model_name": "",
            "n_labels": n_labels, "f1_avg": sk_clean["f1_avg"],
            "dataset": dataset_str, "eval_size": len(texts),
            "model": "sk_tfidf_lr", "tool": "clean", "test_id": "clean",
            "n_eval": len(texts), "n_ok": len(texts),
            "fail_rate": 0.0, "changed_rate": 0.0, "avg_similarity": 1.0,
            "invariance_rate": 1.0,
            "acc": sk_clean["acc"], "f1": sk_clean["f1"],
            "acc_clean": sk_clean["acc"], "f1_clean": sk_clean["f1"],
            "drop_acc": 0.0, "drop_f1": 0.0,
            "time_sec": sk_clean["time_sec"], "peak_rss_mb": sk_clean["peak_rss_mb"],
            "attack_success_rate": "",
            "notes": "peak_delta_mb",
        }

        row_hf = {
            "run_id": run_id, "seed": seed, "repeat_id": repeat_id,
            "dataset_key": dataset_key, "dataset_hf_id": dataset_hf_id, "hf_model_name": hf_model_name,
            "n_labels": n_labels, "f1_avg": hf_clean["f1_avg"],
            "dataset": dataset_str, "eval_size": len(texts),
            "model": "hf_distilbert", "tool": "clean", "test_id": "clean",
            "n_eval": len(texts), "n_ok": len(texts),
            "fail_rate": 0.0, "changed_rate": 0.0, "avg_similarity": 1.0,
            "invariance_rate": 1.0,
            "acc": hf_clean["acc"], "f1": hf_clean["f1"],
            "acc_clean": hf_clean["acc"], "f1_clean": hf_clean["f1"],
            "drop_acc": 0.0, "drop_f1": 0.0,
            "time_sec": hf_clean["time_sec"], "peak_rss_mb": hf_clean["peak_rss_mb"],
            "attack_success_rate": "",
            "notes": "batch_size=32; max_length=256; peak_delta_mb",
        }

        append_row_to_csv(results_csv, row_sk, COLUMNS)
        append_row_to_csv(results_csv, row_hf, COLUMNS)

        print(f"[{dataset_hf_id}] {repeat_id} seed={seed} "
              f"SK acc={sk_clean['acc']:.3f} f1={sk_clean['f1']:.3f} | "
              f"HF acc={hf_clean['acc']:.3f} f1={hf_clean['f1']:.3f}")

    # --- AUGLY
    for test_id, aug_fn in aug_tests.items():
        print(f"[{dataset_hf_id}] AugLy:", test_id)

        for repeat_id in sorted(clean_baselines.keys()):
            seed = clean_baselines[repeat_id]["seed"]
            texts = clean_baselines[repeat_id]["texts"]
            labels = clean_baselines[repeat_id]["labels"]

            sk_base_acc = clean_baselines[repeat_id]["sk"]["acc_clean"]
            sk_base_f1  = clean_baselines[repeat_id]["sk"]["f1_clean"]
            sk_clean_preds_full = clean_baselines[repeat_id]["sk"]["preds_clean"]

            hf_base_acc = clean_baselines[repeat_id]["hf"]["acc_clean"]
            hf_base_f1  = clean_baselines[repeat_id]["hf"]["f1_clean"]
            hf_clean_preds_full = clean_baselines[repeat_id]["hf"]["preds_clean"]

            rss0 = _rss_mb()

            peak_gen = rss0
            t_gen0 = time.time()
            new_texts = [aug_fn(t) for t in texts]
            gen_time = time.time() - t_gen0
            peak_gen = max(peak_gen, _rss_mb())

            stats = calc_similarity_stats(texts, new_texts)

            ok_texts, ok_labels, ok_orig_texts = [], [], []
            ok_sk_clean_preds, ok_hf_clean_preds = [], []

            for orig_t, t_new, y, skp, hfp in zip(texts, new_texts, labels, sk_clean_preds_full, hf_clean_preds_full):
                if isinstance(t_new, str):
                    ok_texts.append(t_new)
                    ok_orig_texts.append(orig_t)
                    ok_labels.append(int(y))
                    ok_sk_clean_preds.append(int(skp))
                    ok_hf_clean_preds.append(int(hfp))

            if len(ok_texts) == 0:
                peak_delta = max(0.0, peak_gen - rss0)

                row_sk = {
                    "run_id": run_id, "seed": seed, "repeat_id": repeat_id,
                    "dataset_key": dataset_key, "dataset_hf_id": dataset_hf_id, "hf_model_name": "",
                    "n_labels": n_labels, "f1_avg": clean_baselines[repeat_id]["sk"]["f1_avg"],
                    "dataset": dataset_str, "eval_size": len(texts),
                    "model": "sk_tfidf_lr", "tool": "augly", "test_id": test_id,
                    **stats,
                    "invariance_rate": "",
                    "acc": "", "f1": "",
                    "acc_clean": sk_base_acc, "f1_clean": sk_base_f1,
                    "drop_acc": "", "drop_f1": "",
                    "time_sec": round(gen_time, 6),
                    "peak_rss_mb": round(peak_delta, 3),
                    "attack_success_rate": "",
                    "notes": "peak_delta_mb; no_ok_texts",
                }

                row_hf = {
                    "run_id": run_id, "seed": seed, "repeat_id": repeat_id,
                    "dataset_key": dataset_key, "dataset_hf_id": dataset_hf_id, "hf_model_name": hf_model_name,
                    "n_labels": n_labels, "f1_avg": clean_baselines[repeat_id]["hf"]["f1_avg"],
                    "dataset": dataset_str, "eval_size": len(texts),
                    "model": "hf_distilbert", "tool": "augly", "test_id": test_id,
                    **stats,
                    "invariance_rate": "",
                    "acc": "", "f1": "",
                    "acc_clean": hf_base_acc, "f1_clean": hf_base_f1,
                    "drop_acc": "", "drop_f1": "",
                    "time_sec": round(gen_time, 6),
                    "peak_rss_mb": round(peak_delta, 3),
                    "attack_success_rate": "",
                    "notes": "peak_delta_mb; no_ok_texts; batch_size=32; max_length=256",
                }

                append_row_to_csv(results_csv, row_sk, COLUMNS)
                append_row_to_csv(results_csv, row_hf, COLUMNS)
                continue

            # SK на ok_texts
            peak_sk = peak_gen
            t_sk0 = time.time()
            sk_preds = sk_model.predict(ok_texts)
            sk_time = time.time() - t_sk0
            peak_sk = max(peak_sk, _rss_mb())

            sk_acc, sk_f1, sk_f1_avg = compute_acc_f1(ok_labels, sk_preds)
            inv_sk = float(np.mean(np.array(ok_sk_clean_preds) == np.array(sk_preds)))

            peak_delta_sk = max(0.0, peak_sk - rss0)

            row_sk = {
                "run_id": run_id, "seed": seed, "repeat_id": repeat_id,
                "dataset_key": dataset_key, "dataset_hf_id": dataset_hf_id, "hf_model_name": "",
                "n_labels": n_labels, "f1_avg": sk_f1_avg,
                "dataset": dataset_str, "eval_size": len(texts),
                "model": "sk_tfidf_lr", "tool": "augly", "test_id": test_id,
                **stats,
                "invariance_rate": inv_sk,
                "acc": sk_acc, "f1": sk_f1,
                "acc_clean": sk_base_acc, "f1_clean": sk_base_f1,
                "drop_acc": float(sk_base_acc - sk_acc),
                "drop_f1": float(sk_base_f1 - sk_f1),
                "time_sec": round(gen_time + sk_time, 6),
                "peak_rss_mb": round(peak_delta_sk, 3),
                "attack_success_rate": "",
                "notes": f"peak_delta_mb; gen+sk_time; n_ok={len(ok_texts)}",
            }
            append_row_to_csv(results_csv, row_sk, COLUMNS)

            # HF на ok_texts
            peak_hf = peak_gen
            t_hf0 = time.time()
            hf_preds, peak_hf = hf_predict_with_peak(hf_model, tokenizer, device, ok_texts, batch_size=32, max_length=256, peak_mb_start=peak_hf)
            hf_time = time.time() - t_hf0

            hf_acc, hf_f1, hf_f1_avg = compute_acc_f1(ok_labels, hf_preds)
            inv_hf = float(np.mean(np.array(ok_hf_clean_preds) == np.array(hf_preds)))

            peak_delta_hf = max(0.0, peak_hf - rss0)

            row_hf = {
                "run_id": run_id, "seed": seed, "repeat_id": repeat_id,
                "dataset_key": dataset_key, "dataset_hf_id": dataset_hf_id, "hf_model_name": hf_model_name,
                "n_labels": n_labels, "f1_avg": hf_f1_avg,
                "dataset": dataset_str, "eval_size": len(texts),
                "model": "hf_distilbert", "tool": "augly", "test_id": test_id,
                **stats,
                "invariance_rate": inv_hf,
                "acc": hf_acc, "f1": hf_f1,
                "acc_clean": hf_base_acc, "f1_clean": hf_base_f1,
                "drop_acc": float(hf_base_acc - hf_acc),
                "drop_f1": float(hf_base_f1 - hf_f1),
                "time_sec": round(gen_time + hf_time, 6),
                "peak_rss_mb": round(peak_delta_hf, 3),
                "attack_success_rate": "",
                "notes": "peak_delta_mb; gen+hf_time; batch_size=32; max_length=256",
            }
            append_row_to_csv(results_csv, row_hf, COLUMNS)

    # --- CHECKLIST
    for test_id, cl_fn in cl_tests.items():
        print(f"[{dataset_hf_id}] CheckList:", test_id)

        for repeat_id in sorted(clean_baselines.keys()):
            seed = clean_baselines[repeat_id]["seed"]
            texts = clean_baselines[repeat_id]["texts"]
            labels = clean_baselines[repeat_id]["labels"]

            sk_base_acc = clean_baselines[repeat_id]["sk"]["acc_clean"]
            sk_base_f1  = clean_baselines[repeat_id]["sk"]["f1_clean"]
            sk_clean_preds_full = clean_baselines[repeat_id]["sk"]["preds_clean"]

            hf_base_acc = clean_baselines[repeat_id]["hf"]["acc_clean"]
            hf_base_f1  = clean_baselines[repeat_id]["hf"]["f1_clean"]
            hf_clean_preds_full = clean_baselines[repeat_id]["hf"]["preds_clean"]

            rss0 = _rss_mb()

            peak_gen = rss0
            t_gen0 = time.time()
            new_texts = [cl_fn(t) for t in texts]
            gen_time = time.time() - t_gen0
            peak_gen = max(peak_gen, _rss_mb())

            stats = calc_similarity_stats(texts, new_texts)

            ok_texts, ok_labels = [], []
            ok_sk_clean_preds, ok_hf_clean_preds = [], []

            for t_new, y, skp, hfp in zip(new_texts, labels, sk_clean_preds_full, hf_clean_preds_full):
                if isinstance(t_new, str):
                    ok_texts.append(t_new)
                    ok_labels.append(int(y))
                    ok_sk_clean_preds.append(int(skp))
                    ok_hf_clean_preds.append(int(hfp))

            if len(ok_texts) == 0:
                peak_delta = max(0.0, peak_gen - rss0)

                row_sk = {
                    "run_id": run_id, "seed": seed, "repeat_id": repeat_id,
                    "dataset_key": dataset_key, "dataset_hf_id": dataset_hf_id, "hf_model_name": "",
                    "n_labels": n_labels, "f1_avg": clean_baselines[repeat_id]["sk"]["f1_avg"],
                    "dataset": dataset_str, "eval_size": len(texts),
                    "model": "sk_tfidf_lr", "tool": "checklist", "test_id": test_id,
                    **stats,
                    "invariance_rate": "",
                    "acc": "", "f1": "",
                    "acc_clean": sk_base_acc, "f1_clean": sk_base_f1,
                    "drop_acc": "", "drop_f1": "",
                    "time_sec": round(gen_time, 6),
                    "peak_rss_mb": round(peak_delta, 3),
                    "attack_success_rate": "",
                    "notes": "peak_delta_mb; no_ok_texts",
                }

                row_hf = {
                    "run_id": run_id, "seed": seed, "repeat_id": repeat_id,
                    "dataset_key": dataset_key, "dataset_hf_id": dataset_hf_id, "hf_model_name": hf_model_name,
                    "n_labels": n_labels, "f1_avg": clean_baselines[repeat_id]["hf"]["f1_avg"],
                    "dataset": dataset_str, "eval_size": len(texts),
                    "model": "hf_distilbert", "tool": "checklist", "test_id": test_id,
                    **stats,
                    "invariance_rate": "",
                    "acc": "", "f1": "",
                    "acc_clean": hf_base_acc, "f1_clean": hf_base_f1,
                    "drop_acc": "", "drop_f1": "",
                    "time_sec": round(gen_time, 6),
                    "peak_rss_mb": round(peak_delta, 3),
                    "attack_success_rate": "",
                    "notes": "peak_delta_mb; no_ok_texts; batch_size=32; max_length=256",
                }

                append_row_to_csv(results_csv, row_sk, COLUMNS)
                append_row_to_csv(results_csv, row_hf, COLUMNS)
                continue

            # SK
            peak_sk = peak_gen
            t_sk0 = time.time()
            sk_preds = sk_model.predict(ok_texts)
            sk_time = time.time() - t_sk0
            peak_sk = max(peak_sk, _rss_mb())

            sk_acc, sk_f1, sk_f1_avg = compute_acc_f1(ok_labels, sk_preds)
            inv_sk = float(np.mean(np.array(ok_sk_clean_preds) == np.array(sk_preds)))

            peak_delta_sk = max(0.0, peak_sk - rss0)

            row_sk = {
                "run_id": run_id, "seed": seed, "repeat_id": repeat_id,
                "dataset_key": dataset_key, "dataset_hf_id": dataset_hf_id, "hf_model_name": "",
                "n_labels": n_labels, "f1_avg": sk_f1_avg,
                "dataset": dataset_str, "eval_size": len(texts),
                "model": "sk_tfidf_lr", "tool": "checklist", "test_id": test_id,
                **stats,
                "invariance_rate": inv_sk,
                "acc": sk_acc, "f1": sk_f1,
                "acc_clean": sk_base_acc, "f1_clean": sk_base_f1,
                "drop_acc": float(sk_base_acc - sk_acc),
                "drop_f1": float(sk_base_f1 - sk_f1),
                "time_sec": round(gen_time + sk_time, 6),
                "peak_rss_mb": round(peak_delta_sk, 3),
                "attack_success_rate": "",
                "notes": f"peak_delta_mb; gen+sk_time; n_ok={len(ok_texts)}",
            }
            append_row_to_csv(results_csv, row_sk, COLUMNS)

            # HF
            peak_hf = peak_gen
            t_hf0 = time.time()
            hf_preds, peak_hf = hf_predict_with_peak(hf_model, tokenizer, device, ok_texts, batch_size=32, max_length=256, peak_mb_start=peak_hf)
            hf_time = time.time() - t_hf0

            hf_acc, hf_f1, hf_f1_avg = compute_acc_f1(ok_labels, hf_preds)
            inv_hf = float(np.mean(np.array(ok_hf_clean_preds) == np.array(hf_preds)))

            peak_delta_hf = max(0.0, peak_hf - rss0)

            row_hf = {
                "run_id": run_id, "seed": seed, "repeat_id": repeat_id,
                "dataset_key": dataset_key, "dataset_hf_id": dataset_hf_id, "hf_model_name": hf_model_name,
                "n_labels": n_labels, "f1_avg": hf_f1_avg,
                "dataset": dataset_str, "eval_size": len(texts),
                "model": "hf_distilbert", "tool": "checklist", "test_id": test_id,
                **stats,
                "invariance_rate": inv_hf,
                "acc": hf_acc, "f1": hf_f1,
                "acc_clean": hf_base_acc, "f1_clean": hf_base_f1,
                "drop_acc": float(hf_base_acc - hf_acc),
                "drop_f1": float(hf_base_f1 - hf_f1),
                "time_sec": round(gen_time + hf_time, 6),
                "peak_rss_mb": round(peak_delta_hf, 3),
                "attack_success_rate": "",
                "notes": "peak_delta_mb; gen+hf_time; batch_size=32; max_length=256",
            }
            append_row_to_csv(results_csv, row_hf, COLUMNS)

    # --- TEXTATTACK
    hf_wrapper = HuggingFaceModelWrapper(hf_model, tokenizer)

    vec = sk_model.named_steps["tfidf"]
    clf = sk_model.named_steps["clf"]
    sk_wrapper = SklearnModelWrapper(clf, TA_VectorizerAdapter(vec))

    attack_sk = TextFoolerJin2019.build(sk_wrapper)
    attack_hf = TextFoolerJin2019.build(hf_wrapper)

    for repeat_id in sorted(clean_baselines.keys()):
        seed = clean_baselines[repeat_id]["seed"]
        texts = clean_baselines[repeat_id]["texts"]
        labels = clean_baselines[repeat_id]["labels"]

        # SK
        rss0 = _rss_mb()
        peak = rss0
        t0 = time.time()

        sk_final, sk_orig, sk_y, sk_skipped, sk_failed, sk_success = run_textattack_for_wrapper(
            attack_sk, texts, labels, N_ATTACK_SK, QUERY_BUDGET
        )

        t_sec = time.time() - t0
        peak = max(peak, _rss_mb())
        peak_delta = max(0.0, peak - rss0)

        if len(sk_final) > 0:
            sk_preds_clean = sk_model.predict(sk_orig)
            sk_acc_clean_sub, sk_f1_clean_sub, sk_f1_avg = compute_acc_f1(sk_y, sk_preds_clean)

            sk_preds_after = sk_model.predict(sk_final)
            sk_acc_after, sk_f1_after, _ = compute_acc_f1(sk_y, sk_preds_after)

            sk_attack_success_rate = (sk_success / (sk_success + sk_failed)) if (sk_success + sk_failed) > 0 else 0.0
            sk_stats = make_stats_for_textattack(sk_orig, sk_final, N_ATTACK_SK, sk_skipped)
        else:
            sk_acc_clean_sub = ""
            sk_f1_clean_sub = ""
            sk_f1_avg = clean_baselines[repeat_id]["sk"]["f1_avg"]
            sk_acc_after = ""
            sk_f1_after = ""
            sk_attack_success_rate = 0.0
            sk_stats = {"n_eval": N_ATTACK_SK, "n_ok": 0, "fail_rate": 1.0, "changed_rate": 0.0, "avg_similarity": 0.0}

        row_sk = {
            "run_id": run_id, "seed": seed, "repeat_id": repeat_id,
            "dataset_key": dataset_key, "dataset_hf_id": dataset_hf_id, "hf_model_name": "",
            "n_labels": n_labels, "f1_avg": sk_f1_avg,
            "dataset": dataset_str, "eval_size": len(texts),
            "model": "sk_tfidf_lr", "tool": "textattack", "test_id": test_id_textattack,
            **sk_stats,
            "invariance_rate": "",
            "acc": sk_acc_after, "f1": sk_f1_after,
            "acc_clean": sk_acc_clean_sub, "f1_clean": sk_f1_clean_sub,
            "drop_acc": (float(sk_acc_clean_sub - sk_acc_after) if sk_acc_after != "" else ""),
            "drop_f1":  (float(sk_f1_clean_sub - sk_f1_after) if sk_f1_after != "" else ""),
            "time_sec": round(t_sec, 6),
            "peak_rss_mb": round(peak_delta, 3),
            "attack_success_rate": float(sk_attack_success_rate),
            "notes": f"peak_delta_mb; N_ATTACK={N_ATTACK_SK}; query_budget={QUERY_BUDGET}; skipped={sk_skipped}; failed={sk_failed}; success={sk_success}",
        }
        append_row_to_csv(results_csv, row_sk, COLUMNS)
        print(f"[{dataset_hf_id}] [repeat {repeat_id} seed={seed}] SK TextAttack ok={row_sk['n_ok']}/{N_ATTACK_SK} ASR={row_sk['attack_success_rate']:.3f}")

        # HF
        rss0 = _rss_mb()
        peak = rss0
        t0 = time.time()

        hf_final, hf_orig, hf_y, hf_skipped, hf_failed, hf_success = run_textattack_for_wrapper(
            attack_hf, texts, labels, N_ATTACK_HF, QUERY_BUDGET
        )

        t_sec = time.time() - t0
        peak = max(peak, _rss_mb())
        peak_delta = max(0.0, peak - rss0)

        if len(hf_final) > 0:
            hf_preds_clean, _ = hf_predict_with_peak(hf_model, tokenizer, device, hf_orig, batch_size=BATCH_SIZE, max_length=MAX_LENGTH)
            hf_acc_clean_sub, hf_f1_clean_sub, hf_f1_avg = compute_acc_f1(hf_y, hf_preds_clean)

            hf_preds_after, _ = hf_predict_with_peak(hf_model, tokenizer, device, hf_final, batch_size=BATCH_SIZE, max_length=MAX_LENGTH)
            hf_acc_after, hf_f1_after, _ = compute_acc_f1(hf_y, hf_preds_after)

            hf_attack_success_rate = (hf_success / (hf_success + hf_failed)) if (hf_success + hf_failed) > 0 else 0.0
            hf_stats = make_stats_for_textattack(hf_orig, hf_final, N_ATTACK_HF, hf_skipped)
        else:
            hf_acc_clean_sub = ""
            hf_f1_clean_sub = ""
            hf_f1_avg = clean_baselines[repeat_id]["hf"]["f1_avg"]
            hf_acc_after = ""
            hf_f1_after = ""
            hf_attack_success_rate = 0.0
            hf_stats = {"n_eval": N_ATTACK_HF, "n_ok": 0, "fail_rate": 1.0, "changed_rate": 0.0, "avg_similarity": 0.0}

        row_hf = {
            "run_id": run_id, "seed": seed, "repeat_id": repeat_id,
            "dataset_key": dataset_key, "dataset_hf_id": dataset_hf_id, "hf_model_name": hf_model_name,
            "n_labels": n_labels, "f1_avg": hf_f1_avg,
            "dataset": dataset_str, "eval_size": len(texts),
            "model": "hf_distilbert", "tool": "textattack", "test_id": test_id_textattack,
            **hf_stats,
            "invariance_rate": "",
            "acc": hf_acc_after, "f1": hf_f1_after,
            "acc_clean": hf_acc_clean_sub, "f1_clean": hf_f1_clean_sub,
            "drop_acc": (float(hf_acc_clean_sub - hf_acc_after) if hf_acc_after != "" else ""),
            "drop_f1":  (float(hf_f1_clean_sub - hf_f1_after) if hf_f1_after != "" else ""),
            "time_sec": round(t_sec, 6),
            "peak_rss_mb": round(peak_delta, 3),
            "attack_success_rate": float(hf_attack_success_rate),
            "notes": f"peak_delta_mb; N_ATTACK={N_ATTACK_HF}; query_budget={QUERY_BUDGET}; skipped={hf_skipped}; failed={hf_failed}; success={hf_success}; batch_size={BATCH_SIZE}; max_length={MAX_LENGTH}",
        }
        append_row_to_csv(results_csv, row_hf, COLUMNS)
        print(f"[{dataset_hf_id}] [repeat {repeat_id} seed={seed}] HF TextAttack ok={row_hf['n_ok']}/{N_ATTACK_HF} ASR={row_hf['attack_success_rate']:.3f}")


    del hf_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return True

for spec in DATASET_SPECS:
    run_one_dataset(spec)

print("\nDONE.")
print("results.csv : ", results_csv)
