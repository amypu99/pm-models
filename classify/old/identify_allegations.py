import pandas as pd
import torch
from transformers import pipeline
import gc
from run_baseline import ministral_setup
from run_questions import label_answers, load_jsonl, questions_setup
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


def prompt_case_chunks(batch, pipe, question, label, tokenizer, results):
    for i, content in enumerate(batch.olmocr_text.values):
        chunked_content = content.split("\n\n")

        all_prompts = [apply_prompt(chunk, question) for chunk in chunked_content]
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
                    "Gold Label": batch[label].iloc[i],
                    "Response": result[0]["generated_text"][1]["content"],
                    "Text": result[0]['generated_text'][0]['content']
                })
                flipped = 1
                break
        if not flipped:
            results.append({
                "Index": batch.Index.iloc[i],
                "Gold Label": batch[label].iloc[i],
                "Response": result[0]["generated_text"][1]["content"]
            })




if __name__ == "__main__":
    # aoe_none_question = "In the text above, is there any mention of prosecutorial misconduct, misconduct by the prosecutor or misconduct by the state? Answer with only a 'Yes' or 'No'.  If you cannot determine the answer, provide your best yes or no guess."
    # identify_allegations(question=aoe_none_question, label = "aoe_none", label_func=label_flipped_answers)

    # aoe_procbar_question = "If there is an alleged assignment of error in the text above, was it procedurally barred? For example, is it barred by res judicata because it was not raised during original trial and now it’s too late? Answer with only a 'Yes' or 'No'.  If you cannot determine the answer, provide your best yes or no guess."
    # identify_allegations(question=aoe_procbar_question, label = "aoe_procbar", label_func=label_answers)

    # aoe_prochist_question = "Is the assignment of error in the procedural history, i.e., if there is prosecutorial misconduct mentioned, was it raised in a previous appeal? Answer with only a 'Yes' or 'No'.  If you cannot determine the answer, provide your best yes or no guess."
    # identify_allegations(question=aoe_prochist_question, label="aoe_prochist", label_func=label_answers)

    aoe_procbar1_question = "Does the text above indicate that the assignments of error were procedurally barred because the appellant filed an untimely appeal? Answer with only a 'Yes' or 'No'.  If you cannot determine the answer, provide your best yes or no guess."
    identify_procbar(jsonl_file = "jsonl/procbar_prochist_olmocr.jsonl", question=aoe_procbar1_question, label="aoe_procbar", label_func=label_answers)

    aoe_procbar2_question = "Does the text above indicate that the assignments of error were procedurally barred because the appellant failed to properly file for appeal? Answer with only a 'Yes' or 'No'.  If you cannot determine the answer, provide your best yes or no guess."
    identify_procbar(jsonl_file="jsonl/dnms_aoe_none_olmocr.jsonl", question=aoe_procbar2_question, label="aoe_procbar", label_func=label_answers)