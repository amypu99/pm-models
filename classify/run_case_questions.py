import pandas as pd
import torch
from transformers import pipeline
import json
import gc
import re
from run_baseline import clean_text, mistral_setup, ministral_setup
import os

os.environ["CUDA_LAUNCH_BLOCKING"]    = "1"
os.environ["CUDA_VISIBLE_DEVICES"]    = "3"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def load_jsonl(filepath):
    data = []
    with open(filepath, 'r') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return pd.DataFrame(data)

# Strict first-token yes/no detector to avoid spurious matches later in the text
YESNO_RE = re.compile(r'^\s*(yes|no)\b', flags=re.IGNORECASE)

def find_whole_word(w):
    return re.compile(r'\b({0})\b'.format(w), flags=re.IGNORECASE).search

def label_answers(answer: str) -> int:
    if not isinstance(answer, str):
        return 99
    head = answer.strip().splitlines()[0] if answer.strip() else ""
    m = YESNO_RE.match(head)
    if not m:
        return 99
    return 1 if m.group(1).lower() == "yes" else 0

def label_flipped_answers(answer: str) -> int:
    lab = label_answers(answer)
    return 99 if lab == 99 else (0 if lab == 1 else 1)

def run_question(question=None, cases_jsonl=None, prompt_func=None, label=None, label_func=None, model=None, tokenizer=None, batch_size=4):
    label1 = label
    if label in ("aoe_procbar1", "aoe_procbar2", "aoe_procbar3", "aoe_procbar4"):
        label1 = "aoe_procbar"

    pipe = pipeline(
        "text-generation",
        model=model,
        torch_dtype=torch.bfloat16,
        tokenizer=tokenizer,
        device_map="auto"
    )

    results = []

    for batch_start in range(0, len(cases_jsonl), batch_size):
        batch_end = min(batch_start + batch_size, len(cases_jsonl))
        batch = cases_jsonl.iloc[batch_start:batch_end]

        prompt_func(
            batch=batch,
            pipe=pipe,
            question=question,
            label=label1,
            tokenizer=tokenizer,
            results=results
        )

        if batch_start % (batch_size * 5) == 0:
            gc.collect()
            torch.cuda.empty_cache()

        print(f"Processed up to sample {batch_end}")

        if batch_start % (batch_size * 10) == 0:
            temp_df = pd.DataFrame(results)
            if "YN" in temp_df.columns:
                temp_df["Predicted Label"] = temp_df["YN"].apply(label_func)
            else:
                temp_df["Predicted Label"] = temp_df["Response"].apply(label_func)
            temp_df.to_csv(f"./results/pipeline_test_2025-09-07/{label}.csv.temp", index=False)

    results_df = pd.DataFrame(results)
    if "YN" in results_df.columns:
        results_df["Predicted Label"] = results_df["YN"].apply(label_func)
    else:
        results_df["Predicted Label"] = results_df["Response"].apply(label_func)
    results_df.to_csv(f"./results/pipeline_test_2025-09-07/{label}.csv", index=False)

    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    return results_df

def prompt_case_head(batch, pipe, question, label, tokenizer, results):
    batch_messages = []

    for content in batch.Context.values:
        cleaned_content = clean_text(content)
        tokenized_content = tokenizer(
            cleaned_content,
            max_length=10000,
            return_tensors='pt',
            truncation=True
        ).to('cuda')
        decoded_content = tokenizer.decode(tokenized_content["input_ids"][0][1:-1])
        full_prompt = (
            f"{decoded_content}\n\n"
            "Above is the appellate case. Read over the case carefully and think step-by-step through "
            f"the following question, answering with only a 'Yes' or 'No'.  If you cannot determine the answer, provide your best yes or no guess: {question}"
        )
        batch_messages.append([{"role": "user", "content": full_prompt}])

    batch_results = pipe(
        batch_messages,
        max_new_tokens=300,
        do_sample=False
    )

    for i, result in enumerate(batch_results):
        results.append({
            "Index": batch.Index.iloc[i],
            "Gold Label": batch[label].iloc[i],
            "Response": result[0]["generated_text"][1]["content"]
        })

def prompt_case_chunks_cited(batch, pipe, question, label, tokenizer, results):
    """
    Ask a yes/no question but REQUIRE the model to back 'Yes' with an exact substring quote.
    Returns JSON {answer, quote, quote_span}. If no substring exists, answer must be 'No' with empty quote.
    """
    batch_messages = []
    decoded_contexts = []

    for ctx in batch.Context.values:
        cleaned = clean_text(ctx)
        toks = tokenizer(cleaned, max_length=8000, return_tensors='pt', truncation=True).to('cuda')
        decoded = tokenizer.decode(toks["input_ids"][0][1:-1])
        decoded_contexts.append(decoded)

        full_prompt = (
            f"{decoded}\n\n"
            "Task: Answer the question strictly from the text ABOVE.\n"
            "Return ONLY this JSON. If you cannot find an exact quote that supports 'Yes', answer 'No' and set quote to \"\".\n"
            "```json\n"
            "{\n"
            "  \"answer\": \"Yes/No\",\n"
            "  \"quote\": \"verbatim substring from the text above (empty if answer is No)\",\n"
            "  \"quote_span\": [start_index_in_text, end_index_in_text]\n"
            "}\n"
            "```\n"
            f"Question: {question}\n"
            "Rules:\n"
            "- The quote MUST be an exact substring of the text above (no paraphrase).\n"
            "- If no exact supporting quote, answer \"No\" and quote=\"\".\n"
        )
        batch_messages.append([{"role": "user", "content": full_prompt}])

    batch_results = pipe(batch_messages, max_new_tokens=220, do_sample=False)

    for i, result in enumerate(batch_results):
        text = result[0]["generated_text"][1]["content"]
        m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, flags=re.DOTALL)
        payload = m.group(1) if m else "{}"
        payload = re.sub(r",\s*}", "}", payload)
        try:
            obj = json.loads(payload)
        except Exception:
            obj = {"answer": "No", "quote": "", "quote_span": [0, 0]}

        quote = (obj.get("quote") or "")
        ctx = decoded_contexts[i]
        # Enforce substring rule
        if quote and quote not in ctx:
            obj["answer"] = "No"
            obj["quote"] = ""
            obj["quote_span"] = [0, 0]

        results.append({
            "Index": batch.Index.iloc[i],
            "Gold Label": batch[label].iloc[i],
            "Response": text,                     # raw model output
            "YN": obj.get("answer", ""),          # clean yes/no for labeling
            "Quote": obj.get("quote", ""),
            "QuoteSpan": obj.get("quote_span", [0, 0]),
        })

def run_ordered_question_pipeline(question, label, q_df, model, tokenizer, batch_size=4):
    dnms_jsonl = load_jsonl("jsonl/dnms.jsonl")
    ms_jsonl = load_jsonl("jsonl/ms.jsonl")

    pipe = pipeline(
        "text-generation",
        model=model,
        torch_dtype=torch.bfloat16,
        tokenizer=tokenizer,
        device_map="auto"
    )

    results = []
    total_rows = len(q_df)
    print(f"Total rows in q_df at start: {total_rows}")

    for batch_start in range(0, total_rows, batch_size):
        batch_end = min(batch_start + batch_size, total_rows)
        batch = q_df.iloc[batch_start:batch_end]

        batch_messages = []
        matched_count = 0

        for idx in batch['Index']:
            matched_row = dnms_jsonl[dnms_jsonl['Index'] == idx]
            if matched_row.empty:
                matched_row = ms_jsonl[ms_jsonl['Index'] == idx]

            if not matched_row.empty:
                matched_count += 1
                context = matched_row['Context'].values[0]
                cleaned_content = clean_text(context)
                tokenized_content = tokenizer(
                    cleaned_content,
                    max_length=18000,
                    return_tensors='pt',
                    truncation=True
                )
                decoded_content = tokenizer.decode(tokenized_content["input_ids"][0][1:-1])
                full_prompt = (
                    f"{decoded_content}\n\n"
                    "Above is the appellate case. Read over the case carefully and think step-by-step through "
                    f"the following question, answering with only a 'Yes' or 'No'. If you are unable to determine the answer, try your best: {question}"
                )
                batch_messages.append([{"role": "user", "content": full_prompt}])

        print(f"Batch {batch_start}–{batch_end}: Matched {matched_count}/{len(batch)} rows")

        if not batch_messages:
            continue

        batch_results = pipe(
            batch_messages,
            max_new_tokens=300,
            do_sample=False
        )

        for i, result in enumerate(batch_results):
            matched_row = dnms_jsonl[dnms_jsonl['Index'] == batch.Index.iloc[i]]
            if matched_row.empty:
                matched_row = ms_jsonl[ms_jsonl['Index'] == batch.Index.iloc[i]]

            results.append({
                "Index": batch.Index.iloc[i],
                "Gold Label": matched_row[label].iloc[0],
                "Response": result[0]["generated_text"][1]["content"]
            })

        if batch_start % (batch_size * 5) == 0:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        print(f"Processed up to sample {batch_end}")

    results_df = pd.DataFrame(results)
    print(f"Total rows processed: {len(results_df)}")
    return results_df

def questions_setup():
    questions = {
        "aoe_procbar1": "Does the text above indicate that the assignments of error were procedurally barred because the"
                        "appellant filed an untimely appeal? Your answer must be a 'Yes' or 'No'.  If you cannot determine"
                        "the answer, provide your best guess.",
        "aoe_procbar2": "Does the text above indicate that the assignments of error were procedurally barred because the "
                        "appellant failed to properly file for appeal? Your answer must be a 'Yes' or 'No'.  If you cannot determine"
                        "the answer, provide your best guess.",
        "aoe_procbar3": "Does the text above indicate that the assignments of error were procedurally barred because the "
                        "appellant failed to present complete and proper documents (eg. fraudulent documents)? Your answer must be a 'Yes' or 'No'.  If you cannot determine"
                        "the answer, provide your best guess.",
        "aoe_procbar4": "Does the text above indicate that the assignments of error were procedurally barred by res judicata?"
                        "If at least one assignment of error is not procedurally barred by res judicata, the answer is 'No'."
                        "Your answer must be a 'Yes' or 'No'.  If you cannot determine the answer, provide your best guess.",
        "case_2001": "What year did the trial take place? The trial comes after the crime and before the indictment and conviction."
                     "Dates listed next to a citation are not relevant. Do NOT respond with the date of a citation."
    }
    return questions
