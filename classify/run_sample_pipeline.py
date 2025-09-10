import os
import sys
import gc
import torch
import pandas as pd

from run_baseline import ministral_setup
from run_case_questions import (
    questions_setup,
    run_question,
    prompt_case_head,
    label_answers
)
from identify_allegations import prompt_case_chunks
from regex_script import identify_regex_dnms  

os.environ["CUDA_LAUNCH_BLOCKING"]    = "1"
os.environ["CUDA_VISIBLE_DEVICES"]    = "6"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def run_pipeline(model_setup, sample_dir, ms_jsonl, dnms_jsonl):
    ms_folder   = os.path.join(sample_dir, "MS")
    dnms_folder = os.path.join(sample_dir, "DNMS")
    if not os.path.isdir(ms_folder) or not os.path.isdir(dnms_folder):
        sys.exit(f"sample_dir must contain 'MS' and 'DNMS' subfolders")
    ms_ids   = {os.path.splitext(f)[0] for f in os.listdir(ms_folder)   if f.lower().endswith(".pdf")}
    dnms_ids = {os.path.splitext(f)[0] for f in os.listdir(dnms_folder) if f.lower().endswith(".pdf")}
    print(f"MS PDFs   found: {sorted(ms_ids)}")
    print(f"DNMS PDFs found: {sorted(dnms_ids)}")
    if not ms_ids and not dnms_ids:
        sys.exit("No PDFs found in sample_cases/MS or sample_cases/DNMS")

    ms_df   = pd.read_json(ms_jsonl,   lines=True)
    dnms_df = pd.read_json(dnms_jsonl, lines=True)

    ms_df   = ms_df[  ms_df["Index"].astype(str).isin(ms_ids)  ].reset_index(drop=True)
    dnms_df = dnms_df[dnms_df["Index"].astype(str).isin(dnms_ids)].reset_index(drop=True)
    print(f"→ {len(ms_df)} MS cases and {len(dnms_df)} DNMS cases will be processed")

    full_df = pd.concat([ms_df, dnms_df], ignore_index=True)
    if full_df.empty:
        sys.exit("No cases to process after filtering JSONLs by sample IDs")
    print(f"Total cases to process: {len(full_df)}")

    regex_df = identify_regex_dnms(sample_dir)
    
    keep_idx = set(
        regex_df.loc[regex_df["Predicted Label"].astype(str) == "0", "Index"].astype(str)
        )    
    filtered_df = full_df[ full_df["Index"].astype(str).isin(keep_idx) ].reset_index(drop=True)
    print(f"After regex filter: {len(filtered_df)} cases remain")
    
    if filtered_df.empty:
        sys.exit("No cases remain after regex filter; exiting")

    gc.collect()
    torch.cuda.empty_cache()
    model, tokenizer = model_setup()

    for q, question_text in questions_setup().items():
        print(f"\nRunning question '{q}' on {len(filtered_df)} cases…")

        if q == "case_2001":
            results_df = run_question(
                question=question_text,
                cases_jsonl=filtered_df,
                prompt_func=prompt_case_head,
                label=q,
                label_func=label_answers,
                model=model,
                tokenizer=tokenizer
            )
        else:
            results_df = run_question(
                question=question_text,
                cases_jsonl=filtered_df,
                prompt_func=prompt_case_chunks,
                label=q,
                label_func=label_answers,
                model=model,
                tokenizer=tokenizer
            )

        if "Response" not in results_df.columns and "Comment" in results_df.columns:
            results_df = results_df.rename(columns={"Comment": "Response"})
        if "Response" not in results_df.columns:
            sys.exit(f"After '{q}', no 'Response' column found in results")

        keep_ids_q = set(
            results_df.loc[results_df["Predicted Label"] == 0, "Index"].astype(str)
        )
        filtered_df = filtered_df[
            filtered_df["Index"].astype(str).isin(keep_ids_q)
        ].reset_index(drop=True)
        print(f"After '{q}' filter: {len(filtered_df)} cases remain")
        if filtered_df.empty:
            print("No cases left—stopping early")
            break

    print("\nPipeline complete.")

if __name__ == "__main__":
    SAMPLE_DIR = "../../cases_pdf_1"
    MS_JSONL   = "../cases_olmocr/MS/ms_olmocr_converted.jsonl"
    DNMS_JSONL = "../cases_olmocr/DNMS/dnms_olmocr_converted.jsonl"

    run_pipeline(ministral_setup, SAMPLE_DIR, MS_JSONL, DNMS_JSONL)
