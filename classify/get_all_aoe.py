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


# def apply_prompt(chunk_text, question):
#     full_prompt = (
#         f"Context information is below.\n---------------------\n{chunk_text}\n---------------------\nGiven the context information and not prior knowledge, answer the query.\nQuery: {question}\nAnswer:"
#     )

#     return full_prompt

def apply_prompt(query, text):
        full_prompt = (
            f"{query}.\n---------------------\n\n\n\n\n—— CASE START ——\n{text}—— CASE END ——"
        )

        return full_prompt


def query_model(pipe, tokenizer, query, text):

    full_prompt = apply_prompt(query, text)
    messages = [{"role": "system", "content": "You are a lawyer. Your job is to read the appellate case (provided) and identify all the allegations of error."},{"role": "user", "content": full_prompt}]
            
    results = pipe(messages, max_new_tokens=3000)
    return results

def full_query(pipe, tokenizer, all_text):
    query = """
            Read the attached legal case and complete the following tasks:

            ────────────────────
            ### TASKS

            1. Count the number of **distinct assignments of error** made by the appellant/defendant.
            2. List each assignment of error clearly and concisely, using numbering or bullet points.
            3. If the assignments are explicitly labeled or numbered (e.g., "Assignment of Error No. 1"), preserve this original numbering and phrasing exactly.

            ────────────────────
            ### OUTPUT FORMAT

            Return **only** this JSON block—nothing else:
            ```json
            {
            "num_errors": <number of distinct assignments of error>,
            "allegation_1": "<full text of first assignment of error>",
            "allegation_2": "<full text of second assignment of error>",
            "allegation_3": "<full text of third assignment of error>",
            ...
            "allegation_n": "<full text of last assignment of error>"
            }
            ```
            """
    tokenized_text = tokenizer(
        all_text,
        max_length=24000,
        return_tensors='pt'
    ).to('cuda')
    decoded_text = tokenizer.decode(tokenized_text["input_ids"][0][1:-1])
    before, found_delimiter, after = decoded_text.rpartition("\n\n")
    generated_text = query_model(pipe, tokenizer, query, before)[0]['generated_text']

    return generated_text[2]['content']




if __name__ == "__main__":
    # from huggingface_hub import login
    # login()
    gc.collect()
    torch.cuda.empty_cache()
    filepath = "../cases_olmocr/remaining.jsonl"
    all_jsonl = load_jsonl(filepath)
    results = {}
    temp_results = {}

    model, tokenizer = ministral_setup()
    pipe = pipeline("text-generation", model=model, max_new_tokens=3000, torch_dtype=torch.bfloat16, device_map='cuda', tokenizer=tokenizer)
    pipe.model = pipe.model.to('cuda')

    for i, row in all_jsonl.iterrows():
        key = row['Index']
        allegation_list = full_query(pipe, tokenizer, row["Context"])
        with open("./results/list_of_allegations_20250527.jsonl", "a") as f:
            json_record = json.dumps({"Index": key, "allegations": allegation_list})
            f.write(json_record + "\n")
        gc.collect()
        torch.cuda.empty_cache()
