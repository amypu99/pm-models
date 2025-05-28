import os
import gc
import torch
import json
import re

from transformers import pipeline
from run_case_questions import load_jsonl
from run_baseline import ministral_setup

#ALLEGATIONS_PATH = "./results/list_of_allegations_20250526.jsonl"
#CASES_PATH       = "../cases_olmocr/all.jsonl"
OUTPUT_PATH      = "./results/extracted_evidence_20250527.jsonl"

ALLEGATIONS_PATH = "./results/test2.jsonl"
CASES_PATH       = "../cases_olmocr/test1.jsonl"

os.environ["CUDA_LAUNCH_BLOCKING"]    = "1"
os.environ["CUDA_VISIBLE_DEVICES"]    = "3"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def clean_json(raw: str) -> str:
    return re.sub(r"^```json\s*|\s*```$", "", raw)

def query_model(pipe, tokenizer, query):

    messages = [{"role": "system", "content": "You are a lawyer. Your job is to read the appellate case (provided) and extract evidence for the allegation of error."},{"role": "user", "content": query}]
            
    results = pipe(messages, max_new_tokens=12000)
    return results

if __name__ == "__main__":
    alle_df  = load_jsonl(ALLEGATIONS_PATH)
    cases_df = load_jsonl(CASES_PATH)

    alle_df.columns  = [c.lower() for c in alle_df.columns]
    cases_df.columns = [c.lower() for c in cases_df.columns]

    model, tokenizer = ministral_setup()
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        max_new_tokens=1200,
    )
    pipe.model = pipe.model.to("cuda")

    open(OUTPUT_PATH, "w").close()

    for i, row in alle_df.iterrows():
        idx = row["index"]  
        raw = row["allegations"]
        cleaned = clean_json(raw)

        try:
            alle_dict = json.loads(cleaned)
        except json.JSONDecodeError:
            print(f"Failed to parse allegations for {idx}.")
            continue

        matched = cases_df[cases_df["index"] == idx]
        if matched.empty:
            print(f"No case found for {idx}.")
            continue
        case_obj = matched.iloc[0].to_dict()

        case_json = json.dumps(matched.iloc[0].to_dict(), ensure_ascii=False)

        for allegation_num, text in alle_dict.items():
            if allegation_num == "num_errors":
                continue
            
            prompt = (
                f"Extract the evidence for this allegation as JSON only.\n\n"
                f"Case: {case_json}\n\n"
                f"ALLEGATION ({allegation_num}): {text}\n\n"
                "Respond with ONLY with this JSON (no trailing commas, all keys & strings double-quoted). Use this exact structure do not add commentary or explanation:\n\n"
                '{\n'
                '  "allegation_num": "<number>",\n'
                '  "allegation": "<text>",\n'
                '  "extracted_text": "<evidence paragraphs>"\n'
                '}\n'
            )

            gen = query_model(pipe, tokenizer, prompt)[0]['generated_text'][2]['content']

            with open(OUTPUT_PATH, "a") as f:
                obj = json.loads(gen)
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

            break

        if i >= 10:
            break
        break

        gc.collect()
        torch.cuda.empty_cache()

    print("Done")
