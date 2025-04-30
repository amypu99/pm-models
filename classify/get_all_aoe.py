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
from run_questions import label_flipped_answers, label_answers, load_jsonl


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


def apply_prompt(chunk_text, question):
    full_prompt = (
        f"Context information is below.\n---------------------\n{chunk_text}\n---------------------\nGiven the context information and not prior knowledge, answer the query.\nQuery: {question}\nAnswer:"
    )

    return full_prompt

def apply_prompt(query, text):
        full_prompt = (
            f"{query}.\n---------------------\n\n\n\n\n—— CASE START ——\n{text}—— CASE END ——"
        )

        return full_prompt


def query_model(model, tokenizer, query, text):
    pipe = pipeline("text-generation", model=model, max_new_tokens=256, torch_dtype=torch.bfloat16, device_map='cuda', tokenizer=tokenizer)
    pipe.model = pipe.model.to('cuda')

    full_prompt = apply_prompt(query, text)
    messages = [{"role": "system", "content": "You are a lawyer. Your job is to read the appellate case (provided) and identify all the allegations of error."},{"role": "user", "content": full_prompt}]
            
    results = pipe(messages, max_new_tokens=256)
    return results


if __name__ == "__main__":
    from huggingface_hub import login
    login()
    gc.collect()
    torch.cuda.empty_cache()
    filepath = "../cases_olmocr/DNMS/dnms_olmocr_leftover.jsonl"
    all_jsonl = load_jsonl(filepath)
    results = {}
    temp_results = {}

    model, tokenizer = ministral_setup()
    for i, key in enumerate([x['Source-File'].replace(".pdf", "") for x in all_jsonl.metadata]):
        query = """
        Read the attached legal case and complete the following tasks:

        ────────────────────
        ### 1. Name and list all the assignments of error or allegations claimed by the appellant/defendant.

        ────────────────────
        ### 2. Output format
        Return **only** this JSON block—nothing else:
        ```json
        {
        "allegation_1": "<full text of assignment of error>",
        "allegation_2": "<full text of second assignment of error>"
        "allegation_3": "<full text of third assignment of error>"
        ...
        "allegation_n": "<full text of last assignment of error>"
        }
        ```
        """
        all_text = all_jsonl.text[i]
        tokenized_text = tokenizer(
            all_text,
            max_length=20000,
            return_tensors='pt'
        ).to('cuda')
        decoded_text = tokenizer.decode(tokenized_text["input_ids"][0][1:-1])
        before, found_delimiter, after = all_text.rpartition("\n\n")
        # print("\n\nlen")
        generated_text = query_model(model, tokenizer, query, before)[0]['generated_text']
        # print(len(generated_text))
        # print("\n\ngenerated text")
        # print(generated_text)
        # print(query_model(model, tokenizer, query, before)[0]['generated_text'][2]['content'])
        results[key] = generated_text[2]['content']
        # print("\n\nresults")
        # print(results[key])
        with open("list_of_allegations.jsonl", "a") as f:
            json_record = json.dumps({"index": key, "allegations": results[key]})
            f.write(json_record + "\n")
        gc.collect()
        torch.cuda.empty_cache()
