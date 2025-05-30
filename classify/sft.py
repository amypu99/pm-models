import torch
import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from trl import SFTConfig, SFTTrainer
import wandb
from run_case_questions import load_jsonl
from datasets import Dataset, load_dataset

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


train_df = load_jsonl("../cases_olmocr/train.jsonl").sample(frac=1).reset_index(drop=True)
test_df = load_jsonl("../cases_olmocr/test.jsonl").sample(frac=1).reset_index(drop=True)

# print(train_df)


train_data = train_df["Context"].values.tolist()
train_labels = list(map(lambda x: "meet standards" if x == 1 else "does not meet standards", train_df["meets_standards"].values.tolist()))
test_data = test_df["Context"].values.tolist()
test_labels = list(map(lambda x: "meet standards" if x == 1 else "does not meet standards", test_df["meets_standards"].values.tolist()))


train_data_with_labels = Dataset.from_dict({"text": train_data, "labels": train_labels})
test_data_with_labels = Dataset.from_dict({"text": test_data, "labels": test_labels})

dataset_train = train_data_with_labels.map(
    lambda x: {"messages": [{"role": "system", "content": "You are a lawyer. Your job is to read the legal case (provided) and determine if it meets the standards for further review of prosecutorial misconduct"},
              {"role": "user", "content": x["text"]},
              {"role": "assistant", "content": str(x["labels"])}]},
    remove_columns=train_data_with_labels.column_names,
)
print("TRAIN SET", dataset_train)
# print(dataset_train["messages"][0])

dataset_test = test_data_with_labels.map(
    lambda x: {"messages": [{"role": "system", "content": "You are a lawyer. Your job is to read the legal case (provided) and determine if it meets the standards for further review of prosecutorial misconduct"},
              {"role": "user", "content": x["text"]},
              {"role": "assistant", "content": str(x["labels"])}]},
    remove_columns=test_data_with_labels.column_names,
)

print("TEST SET", dataset_test)
# print(dataset_test["messages"][0])

# https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.TrainingArguments
# https://huggingface.co/docs/trl/en/sft_trainer#trl.SFTConfig
training_args = SFTConfig(
    # max_length=32768,
    # output_dir="/home/sky/sky_workdir/models/sft-llama-cot",
    output_dir="./sft-out",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    learning_rate=1e-05,
    lr_scheduler_type="cosine",
    eval_strategy="steps",
    eval_steps=0.20,
    save_strategy="epoch",
    warmup_ratio=0.05,
    num_train_epochs=1,
    gradient_accumulation_steps=1,
    fp16=True,
    dataloader_pin_memory=True,
    gradient_checkpointing=True,
    logging_first_step=True,
    logging_steps=1,
    report_to="wandb",
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype=torch.bfloat16,  # use bfloat16 for training
)


# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    use_fast=True,  # Optional but recommended
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset_train,
    args=training_args,
    eval_dataset=dataset_test,
    # processing_class=tokenizer,
)
trainer.train()