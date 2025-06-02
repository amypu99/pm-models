import os
import gc
import torch
import json
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline
)
from datasets import load_dataset
from peft import LoraConfig, PeftModel


from run_baseline import clean_text, mistral_setup, ministral_setup, llama_setup
from run_case_questions import label_flipped_answers, label_answers, load_jsonl


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'



def query_model(pipe, tokenizer, text):

    messages = [{"role": "system", "content": "You are a lawyer. Your job is to read the legal case (provided) and determine if it meets the standards for further review of prosecutorial misconduct"},{"role": "user", "content": text}]

    results = pipe(messages, max_new_tokens=256)
    return results

def full_query(pipe, tokenizer, all_text):

    generated_text = query_model(pipe, tokenizer, all_text)[0]['generated_text']
    print(generated_text)

    return generated_text[2]['content']




if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    filepath = "../cases_olmocr/test.jsonl"
    all_jsonl = load_jsonl(filepath)
    results = {}
    temp_results = {}

    model_name = "../../my-cool-model/checkpoint-111"

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Load tokenizer
    # tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct",use_fast=True)
    tokenizer = AutoTokenizer.from_pretrained("../../my-cool-model/checkpoint-111")

    pipe = pipeline("text-generation", model=model, max_new_tokens=256, torch_dtype=torch.bfloat16, device_map='cuda', tokenizer=tokenizer)
    pipe.model = pipe.model.to('cuda')

    for i, row in all_jsonl.iterrows():
        key = row['Index']
        result = full_query(pipe, tokenizer, row["Context"])
        with open("./results/sft_results.jsonl", "a") as f:
            json_record = json.dumps({"index": key, "prediction": result})
            f.write(json_record + "\n")
        gc.collect()
        torch.cuda.empty_cache()
