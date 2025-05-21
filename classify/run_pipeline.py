from run_baseline import ministral_setup
from run_questions import *
from identify_allegations import prompt_case_chunks
from regex import identify_regex_dnms


def run_pipeline(model_setup, pdf_dir, ms_jsonl, dnms_jsonl):
    questions_dict = questions_setup()
    regex_results = identify_regex_dnms(pdf_dir)

    gc.collect()
    torch.cuda.empty_cache()
    model, tokenizer = model_setup()

    full_jsonl = combine_jsonl(ms_jsonl, dnms_jsonl)
    # full_jsonl = load_jsonl("jsonl/dnms_aoe_none_olmocr.jsonl")
    # full_jsonl =full_jsonl.sample(n=5, random_state=42)
    filtered_jsonl = filter_jsonl(regex_results, full_jsonl)

    results_df = pd.DataFrame()
    # For each case-specific question, case_2001, and whether untimely or improper paperwork
    for q in questions_dict:
        print(f"\nRunning question: {q}")
        # Run case_2001 without chunking
        if q =="case_2001":
            results_df = run_question(question=questions_dict[q], cases_jsonl=filtered_jsonl,
                                      prompt_func=prompt_case_head, label=q, label_func=label_answers, model=model,
                                      tokenizer=tokenizer)
        # Run aoe_procbar with chunking
        else:
            results_df = run_question(question=questions_dict[q], cases_jsonl=filtered_jsonl,
                                      prompt_func=prompt_case_chunks, label=q, label_func=label_answers, model=model,
                                      tokenizer=tokenizer)
        # Filter rows where Response Label == 0 (MS) to run the next iteration
        filtered_jsonl = filter_jsonl(results_df, filtered_jsonl)
        filtered_df_size = len(filtered_jsonl)
        print(f"New q_df size after filtering: {filtered_df_size}")



def combine_jsonl(dnms_jsonl_fp, ms_jsonl_fp):
    dnms_jsonl = pd.read_json(dnms_jsonl_fp, lines=True)
    ms_jsonl = pd.read_json(ms_jsonl_fp, lines=True)
    # dnms_sample = dnms_jsonl.sample(n=5, random_state=42)
    # ms_sample = ms_jsonl.sample(n=5, random_state=42)
    # combined_jsonl = pd.concat([dnms_sample, ms_sample]).reset_index(drop=True)
    combined_jsonl = pd.concat([dnms_jsonl, ms_jsonl])

    return combined_jsonl


def filter_jsonl(df, jsonl_df):
    # Take in csv, filter just the cases with 0 (MS so far)
    # Based on index of those columns, filter jsonl
    print("columns", df.columns)
    filtered_df = df[df['Predicted Label'] == 0]
    filtered_idx = filtered_df["Index"].astype(str).tolist()
    filtered_json = jsonl_df[jsonl_df['Index'].astype(str).isin(filtered_idx)].to_dict('records')
    print(f"Filtered JSONL written")

    return pd.DataFrame(filtered_json)

if __name__ == "__main__":
    ms_path = "../cases_olmocr/MS/ms_olmocr_converted.jsonl"
    dnms_path = "../cases_olmocr/DNMS/dnms_olmocr_converted.jsonl"
    run_pipeline(ministral_setup, "../cases_pdf", ms_path, dnms_path)