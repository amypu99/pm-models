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
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

CHUNK_LENGTH = 2000

def apply_prompt(chunk_text):
    question = "In this paragraph, is there any mention of prosecutorial misconduct, misconduct by the prosecutor, misconduct by the state, or Brady Violation?"

    full_prompt = (
        f"Read the following paragraph from an Ohio appellate case carefully and think step-by-step through "
        f"the following question, answering with only a 'Yes' or 'No'.  If you cannot determine the answer, provide your best yes or no guess: {question}\n\nParagraph from the case: {chunk_text}"
    )

    return full_prompt


def identify_allegations(batch_size=1):
    gc.collect()
    torch.cuda.empty_cache()
    model, tokenizer = ministral_setup()

    question_jsonl = load_jsonl("another_example.jsonl")

    pipe = pipeline(
        "text-generation",
        model=model,
        torch_dtype=torch.bfloat16,
        # device_map='auto',
        device_map='cuda',
        tokenizer=tokenizer
    )
    pipe.model = pipe.model.to('cuda')

    results = []

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    for batch_start in range(0, 100, batch_size):
        batch_end = min(batch_start + batch_size, len(question_jsonl))
        batch = question_jsonl.iloc[batch_start:batch_end]

        batch_messages = []

        for i, content in enumerate(batch.Context.values):
            cleaned_content = clean_text(content)
            chunked_content = cleaned_content.split("\u00b6")
            print([len(x) for x in chunked_content])
            # print('\n\n\n\n\n\n\n'.join(chunked_content))
            
            all_prompts = [apply_prompt(chunk) for chunk in chunked_content]
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

            label = "aoe_none"
            flipped = 0
            for k, result in enumerate(example_outputs):
                if label_answers(result[0]["generated_text"][1]["content"]) == 1:
                    results.append({
                        "Index": batch.Index.iloc[i],
                        "Response": result[0]["generated_text"][1]["content"],
                        # "Text": chunk,
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
                temp_df["Response Label"] = temp_df["Response"].apply(label_flipped_answers)
                temp_df.to_csv(f"./mistral_questions/{label}.csv.temp", index=False)

    # results_df = pd.DataFrame(results)

    # results_df["Response Label"] = results_df["Response"].apply(label_flipped_answers)
    # results_df.to_csv(f"./mistral_questions/{label}.csv", index=False)
    # gc.collect()
    # torch.cuda.empty_cache()
    # if torch.cuda.is_available():
    #     torch.cuda.synchronize()


if __name__ == "__main__":
    from huggingface_hub import login
    login()
    identify_allegations()