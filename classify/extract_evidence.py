import os
import gc
import torch
import json
import re

from transformers import pipeline
from run_case_questions import load_jsonl
from run_baseline import ministral_setup

ALLEGATIONS_PATH = "./results/list_of_allegations_20250526.jsonl"
CASES_PATH       = "../cases_olmocr/all.jsonl"
OUTPUT_PATH      = "./results/extracted_evidence_20250527.jsonl"

# ALLEGATIONS_PATH = "./results/test2.jsonl"
# CASES_PATH       = "../cases_olmocr/test1.jsonl"

os.environ["CUDA_LAUNCH_BLOCKING"]    = "1"
os.environ["CUDA_VISIBLE_DEVICES"]    = "3"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def clean_json(raw: str) -> str:
    return re.sub(r"^```json\s*|\s*```$", "", raw)

def apply_prompt(query, text):
        full_prompt = (
            f"{query}.\n---------------------\n\n\n\n\n—— CASE START ——\n{text}—— CASE END ——"
        )

        return full_prompt

def query_model(pipe, tokenizer, query, text):

    full_prompt = apply_prompt(query, text)
    messages = [{"role": "system", "content": "You are a lawyer. Your job is to read the appellate case (provided) and extract the evidence for the allegation of error."},{"role": "user", "content": full_prompt}]
            
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
        max_new_tokens=12000,
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

        for allegation_num, text in alle_dict.items():
            if allegation_num == "num_errors":
                continue

            query = """
            Read the attached legal case and complete the following tasks:

            ────────────────────
            ### TASKS

            1. Read the attached legal case and find the assignment of error that's mentioned below in the case. 
            2. Extract the corresponding discussion and evidence paragraphs regarding this assignment of error. 
            3. "Discussion" refers to the full set of paragraphs typically following the assignment title that elaborate on the nature of the alleged error, the court's reasoning, any citations, and its ruling on the issue. It typically begins with a paragraph referencing the assignment number and ends just before the next assignment of error or heading.

            ────────────────────
            ### OUTPUT FORMAT

            Return **only** this JSON block—nothing else:
            ```json
            {
            "discussion": <full corresponding evidence and discussion regarding assignment of error>,
            }
            ```
            """

            tokenized_text = tokenizer(
                matched["context"].values[0],
                max_length=24000,
                return_tensors='pt'
            ).to('cuda')
            decoded_text = tokenizer.decode(tokenized_text["input_ids"][0][1:-1])
            before, found_delimiter, after = decoded_text.rpartition("\n\n")
            extracted_discussion = query_model(pipe, tokenizer, query, before)[0]['generated_text'][2]['content']

            with open(OUTPUT_PATH, "a") as f:
                extracted_discussion_dict = {"index": idx}
                extracted_discussion_dict["allegation_num"] = allegation_num
                extracted_discussion_dict["allegation"] = text
                extracted_discussion_dict["discussion"] = extracted_discussion.strip()
                f.write(json.dumps(extracted_discussion_dict, ensure_ascii=False) + "\n")

        gc.collect()
        torch.cuda.empty_cache()

    print("Done")