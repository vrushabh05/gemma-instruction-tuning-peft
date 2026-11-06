"""Part I: instruction-tune Gemma-2-2b-it with 4-bit QLoRA on 3,000 Dolly-15K examples (response-only loss),
then generate answers on 100 held-out Dolly prompts (in-distribution) and 100 Alpaca prompts (out-of-distribution).

Requires: transformers==4.51.3, trl==0.8.6, peft, bitsandbytes, datasets.
"""
import pandas as pd
import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

from common import BNB_CONFIG, DEVICE, HF_TOKEN, SEED

MODEL = "google/gemma-2-2b-it"
OUT = "gemma2_dolly_sft"

dolly = load_dataset("databricks/databricks-dolly-15k", split="train").shuffle(seed=SEED)
train, test_id = dolly.select(range(3000)), dolly.select(range(3000, 3200))


def fmt(s):
    ctx = f"\nContext: {s['context']}" if s.get("context", "").strip() else ""
    return {"text": f"### Instruction:\n{s['instruction']}{ctx}\n\n### Response:\n{s['response']}"}


train_f = train.map(fmt, remove_columns=train.column_names)
test_f = test_id.map(fmt, remove_columns=test_id.column_names)

tok = AutoTokenizer.from_pretrained(MODEL, token=HF_TOKEN)
tok.pad_token = tok.pad_token or tok.eos_token
tok.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=BNB_CONFIG, device_map="auto",
                                             token=HF_TOKEN, attn_implementation="eager")
model.config.use_cache = False
model = prepare_model_for_kbit_training(model)

# Loss only on the response: everything before "### Response:" is masked to -100.
collator = DataCollatorForCompletionOnlyLM(
    response_template=tok.encode("### Response:", add_special_tokens=False),
    instruction_template=tok.encode("### Instruction:", add_special_tokens=False),
    tokenizer=tok,
)

lora = LoraConfig(task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])

# NOTE: get_peft_model wraps and modifies `model` in place. Reload a fresh base model before generating
# "base" responses, or the comparison below silently uses the adapter-injected model.
peft_model = get_peft_model(model, lora)
peft_model.print_trainable_parameters()  # 20,766,720 trainable of 2,635,108,608 (0.79%)

args = TrainingArguments(output_dir=OUT, num_train_epochs=3, per_device_train_batch_size=4,
                         gradient_accumulation_steps=4, warmup_ratio=0.03, learning_rate=2e-4, bf16=True,
                         logging_steps=50, save_strategy="epoch", eval_strategy="epoch",
                         load_best_model_at_end=True, metric_for_best_model="loss", report_to="none",
                         seed=SEED, optim="paged_adamw_8bit", lr_scheduler_type="cosine")
trainer = SFTTrainer(model=peft_model, args=args, train_dataset=train_f, eval_dataset=test_f, tokenizer=tok,
                     data_collator=collator, max_seq_length=1024, packing=False, dataset_text_field="text")
trainer.train()
trainer.save_model(OUT)


def generate(m, instruction, context="", max_new_tokens=80):
    ctx = f"\nContext: {context}" if context.strip() else ""
    prompt = f"### Instruction:\n{instruction}{ctx}\n\n### Response:\n"
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        out = m.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0], skip_special_tokens=True).split("### Response:")[-1].strip()


fresh_base = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=BNB_CONFIG, device_map="auto",
                                                  token=HF_TOKEN, attn_implementation="eager")
fine_tuned = PeftModel.from_pretrained(fresh_base, OUT).eval()  # adapter on, base weights shared

rows = []
for s in test_id.select(range(100)):
    rows.append(dict(set="ID", instruction=s["instruction"], reference=s["response"],
                     ft_response=generate(fine_tuned, s["instruction"], s.get("context", ""))))
for s in load_dataset("tatsu-lab/alpaca", split="train").shuffle(seed=SEED).select(range(100)):
    rows.append(dict(set="OOD", instruction=s["instruction"], reference=s["output"],
                     ft_response=generate(fine_tuned, s["instruction"], s.get("input", ""))))
pd.DataFrame(rows).to_csv("part1_responses.csv", index=False)
