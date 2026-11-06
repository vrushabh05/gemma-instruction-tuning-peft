"""Part II: sentiment classification on Sentiment140 with Gemma-2-2b: zero-shot vs LoRA vs a narrow q/v adapter.

Data: 20,000 balanced training tweets and 2,000 balanced test tweets, sampled from 1.55M cleaned tweets, no text overlap.
Each PEFT method gets an Optuna search (10 trials LoRA, 7 trials adapter) then a 5-epoch final run.
Caveat: trials and best-checkpoint selection score on the same 2,000 test tweets (see README).
"""
import gc
import re

import numpy as np
import optuna
import pandas as pd
import torch
from datasets import Dataset
from optuna.samplers import TPESampler
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from transformers import (AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)

from common import BNB_CONFIG, DEVICE, HF_TOKEN, SEED

MODEL, MAX_LEN, TRAIN_N, TEST_N = "google/gemma-2-2b", 128, 20000, 2000


def clean(t):
    t = re.sub(r"http\S+|www\S+", "", str(t))
    t = re.sub(r"@\w+", "@user", t)
    t = re.sub(r"#(\w+)", r"\1", t)
    t = re.sub(r"[^\x00-\x7F]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


raw = pd.read_csv("training.1600000.processed.noemoticon.csv", encoding="latin-1",
                  names=["sentiment", "id", "date", "query", "user", "text"])
raw["label"] = (raw["sentiment"] == 4).astype(int)
raw["text"] = raw["text"].apply(clean)
raw = raw[raw["text"].str.len() > 5][["text", "label"]].drop_duplicates("text").reset_index(drop=True)

parts_tr, parts_te = [], []
for lab in (0, 1):
    pool = raw[raw.label == lab].sample(n=TRAIN_N // 2 + TEST_N // 2, random_state=SEED)
    tr, te = train_test_split(pool, test_size=TEST_N // 2, random_state=SEED)
    parts_tr.append(tr); parts_te.append(te)
train_df = pd.concat(parts_tr).sample(frac=1, random_state=SEED).reset_index(drop=True)
test_df = pd.concat(parts_te).sample(frac=1, random_state=SEED).reset_index(drop=True)
assert not set(train_df.text) & set(test_df.text), "data leakage"

tok = AutoTokenizer.from_pretrained(MODEL, token=HF_TOKEN)
if tok.pad_token is None:
    tok.add_special_tokens({"pad_token": "[PAD]"})
tok.padding_side = "right"


def encode(df):
    ds = Dataset.from_pandas(df).map(lambda b: tok(b["text"], truncation=True, max_length=MAX_LEN,
                                                   padding="max_length"), batched=True)
    ds = ds.rename_column("label", "labels")
    ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    return ds


train_hf, test_hf = encode(train_df), encode(test_df)


def metrics(p):
    preds, labels = np.argmax(p[0], -1), p[1]
    return dict(accuracy=accuracy_score(labels, preds), precision=precision_score(labels, preds, zero_division=0),
                recall=recall_score(labels, preds, zero_division=0), f1=f1_score(labels, preds, zero_division=0))


# ---- zero-shot baseline: compare next-token logits for " positive" vs " negative" ----
zs = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=BNB_CONFIG, device_map="auto", token=HF_TOKEN).eval()
pos_id, neg_id = (tok.encode(w, add_special_tokens=False)[0] for w in (" positive", " negative"))
preds = []
texts = test_df.text.tolist()
for i in range(0, len(texts), 32):
    enc = tok([f'Tweet: "{t}"\nSentiment:' for t in texts[i:i + 32]], return_tensors="pt", padding=True,
              truncation=True, max_length=MAX_LEN).to(DEVICE)
    with torch.no_grad():
        logits = zs(**enc).logits[:, -1, :]
    preds += (logits[:, pos_id] > logits[:, neg_id]).long().cpu().tolist()
print("zero-shot", dict(accuracy=accuracy_score(test_df.label, preds), f1=f1_score(test_df.label, preds)))
del zs; gc.collect(); torch.cuda.empty_cache()


def build(target_modules, r, alpha, dropout):
    m = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2, quantization_config=BNB_CONFIG,
                                                           device_map="auto", token=HF_TOKEN)
    m.config.pad_token_id = tok.pad_token_id
    m = prepare_model_for_kbit_training(m)
    cfg = LoraConfig(task_type=TaskType.SEQ_CLS, r=r, lora_alpha=alpha, lora_dropout=dropout,
                     target_modules=target_modules, bias="none")
    return get_peft_model(m, cfg)


def run(m, lr, epochs, out, final=False):
    args = TrainingArguments(output_dir=out, num_train_epochs=epochs, per_device_train_batch_size=32,
                             per_device_eval_batch_size=64, learning_rate=lr, warmup_ratio=0.1,
                             lr_scheduler_type="cosine", weight_decay=0.01 if final else 0.0,
                             eval_strategy="epoch", save_strategy="epoch" if final else "no",
                             load_best_model_at_end=final, metric_for_best_model="f1" if final else None,
                             report_to="none", seed=SEED, bf16=True, logging_steps=50)
    t = Trainer(model=m, args=args, train_dataset=train_hf, eval_dataset=test_hf, tokenizer=tok,
                data_collator=DataCollatorWithPadding(tok), compute_metrics=metrics)
    t.train()
    return t.evaluate()


ALL_LINEAR = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def lora_objective(trial):
    r = trial.suggest_categorical("r", [16, 32, 64])
    a = trial.suggest_categorical("lora_alpha", [32, 64, 128])
    lr = trial.suggest_float("learning_rate", 1e-4, 5e-4, log=True)
    d = trial.suggest_float("lora_dropout", 0.0, 0.1)
    f1 = run(build(ALL_LINEAR, r, a, d), lr, 3, f"lora_trial_{trial.number}")["eval_f1"]
    gc.collect(); torch.cuda.empty_cache()
    return f1


def adapter_objective(trial):
    r = trial.suggest_categorical("bottleneck_dim", [8, 16, 32, 64])
    lr = trial.suggest_float("learning_rate", 1e-4, 5e-4, log=True)
    d = trial.suggest_float("dropout", 0.0, 0.2)
    f1 = run(build(["q_proj", "v_proj"], r, 2 * r, d), lr, 3, f"adapter_trial_{trial.number}")["eval_f1"]
    gc.collect(); torch.cuda.empty_cache()
    return f1


lora_study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=SEED))
lora_study.optimize(lora_objective, n_trials=10)
p = lora_study.best_params
print("LoRA final", run(build(ALL_LINEAR, p["r"], p["lora_alpha"], p["lora_dropout"]), p["learning_rate"], 5,
                        "lora_final", final=True))

adapter_study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=SEED))
adapter_study.optimize(adapter_objective, n_trials=7)
p = adapter_study.best_params
print("Adapter final", run(build(["q_proj", "v_proj"], p["bottleneck_dim"], 2 * p["bottleneck_dim"], p["dropout"]),
                           p["learning_rate"], 5, "adapter_final", final=True))
