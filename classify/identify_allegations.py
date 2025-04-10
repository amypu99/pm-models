import pandas as pd
import torch
from transformers import pipeline
import json
import gc
import re
from run_baseline import clean_text, mistral_setup, ministral_setup, llama_setup
from run_questions import label_flipped_answers, label_answers, load_jsonl
import math
import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

CHUNK_LENGTH = 2000

def apply_prompt(chunk_text, question):
    full_prompt = (
        f"Context information is below.\n---------------------\n{chunk_text}\n---------------------\nGiven the context information and not prior knowledge, answer the query.\nQuery: {question}\nAnswer:"
    )

    return full_prompt


def identify_allegations(batch_size=4, question=None, label=None, label_func=None):
    gc.collect()
    torch.cuda.empty_cache()
    model, tokenizer = ministral_setup()


    question_jsonl = load_jsonl("dnms_aoe_none_olmocr.jsonl")

    pipe = pipeline(
        "text-generation",
        model=model,
        torch_dtype=torch.bfloat16,
        # device_map='auto',
        device_map='cuda',
        tokenizer=tokenizer
    )
    # pipe.model = pipe.model.to('cuda')

    results = []

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # print(len(question_jsonl))

    for batch_start in range(0, len(question_jsonl), batch_size):
        batch_end = min(batch_start + batch_size, len(question_jsonl))
        batch = question_jsonl.iloc[batch_start:batch_end]

        batch_messages = []

        for i, content in enumerate(batch.olmocr_text.values):
            # cleaned_content = clean_text(content)
            chunked_content = content.split("\n\n")
            
            all_prompts = [apply_prompt(chunk, question) for chunk in chunked_content]
            # print("\n\n\n".join(all_prompts))
            all_messages = [[{"role": "user", "content": x}] for x in all_prompts]


            num_chunks = min(5, len(chunked_content))

            example_outputs = []
            for j in range(0, len(all_messages), num_chunks):
                start = j
                end = min(j + num_chunks, len(all_messages))
                batch_results = pipe(
                    all_messages[start:end],
                    max_new_tokens=300,
                    do_sample=False
                )
                example_outputs += batch_results

            flipped = 0
            for k, result in enumerate(example_outputs):
                if label_answers(result[0]["generated_text"][1]["content"]) == 1:
                    results.append({
                        "Index": batch.Index.iloc[i],
                        "Response": result[0]["generated_text"][1]["content"],
                        "Text": result[0]['generated_text'][0]['content'],
                        label: batch[label].iloc[i]
                    })
                    flipped = 1
                    break
            if not flipped:
                results.append({
                    "Index": batch.Index.iloc[i],
                    "Response": result[0]["generated_text"][1]["content"],
                    label: batch[label].iloc[i]
                })

            if batch_start % (batch_size * 5) == 0:
                gc.collect()
                torch.cuda.empty_cache()

            print(f"Processed up to sample {batch_end}")

            if batch_start % (batch_size * 10) == 0:
                temp_df = pd.DataFrame(results)
                temp_df["Response Label"] = temp_df["Response"].apply(label_func)
                temp_df.to_csv(f"./ministral_olmocr_questions/{label}.csv.temp", index=False)

    results_df = pd.DataFrame(results)

    results_df["Response Label"] = results_df["Response"].apply(label_func)
    results_df.to_csv(f"./ministral_olmocr_questions/{label}.csv", index=False)
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


if __name__ == "__main__":
    # aoe_none_question = "In the text above, is there any mention of prosecutorial misconduct, misconduct by the prosecutor or misconduct by the state? Answer with only a 'Yes' or 'No'.  If you cannot determine the answer, provide your best yes or no guess."
    # identify_allegations(question=aoe_none_question, label = "aoe_none", label_func=label_flipped_answers)

    # aoe_procbar_question = "If there is an alleged assignment of error in the text above, was it procedurally barred? For example, is it barred by res judicata because it was not raised during original trial and now it’s too late? Answer with only a 'Yes' or 'No'.  If you cannot determine the answer, provide your best yes or no guess."
    # identify_allegations(question=aoe_procbar_question, label = "aoe_procbar", label_func=label_answers)

    # aoe_prochist_question = "Is the assignment of error in the procedural history, i.e., if there is prosecutorial misconduct mentioned, was it raised in a previous appeal? Answer with only a 'Yes' or 'No'.  If you cannot determine the answer, provide your best yes or no guess."
    # identify_allegations(question=aoe_prochist_question, label="aoe_prochist", label_func=label_answers)

    aoe_noneprocbar_question = "In the text above, is there any mention of prosecutorial misconduct, misconduct by the prosecutor or misconduct by the state that is not not barred by res judicata? Answer with only a 'Yes' or 'No'.  If you cannot determine the answer, provide your best yes or no guess."
    identify_allegations(question=aoe_noneprocbar_question, label="aoe_none", label_func=label_answers)