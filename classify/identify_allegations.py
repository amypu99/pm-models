import pandas as pd
import torch
from transformers import pipeline
import json
import gc
import re
from run_baseline import clean_text, mistral_setup, ministral_setup, llama_setup
from run_questions import label_flipped_answers, label_answers, load_jsonl
import math

CHUNK_LENGTH = 2000


def identify_allegations(batch_size=1):
    gc.collect()
    torch.cuda.empty_cache()
    model, tokenizer = mistral_setup()

    question_jsonl = load_jsonl("dnms.jsonl")

    pipe = pipeline(
        "text-generation",
        model=model,
        torch_dtype=torch.bfloat16,
        device_map='cuda',
        tokenizer=tokenizer
    )
    pipe.model = pipe.model.to('cuda')

    results = []

    # counter = 0
    for batch_start in range(0, 100, batch_size):
        batch_end = min(batch_start + batch_size, len(question_jsonl))
        batch = question_jsonl.iloc[batch_start:batch_end]

        batch_messages = []
        counter = 0
        for content in batch.Context.values:
            cleaned_content = clean_text(content)
            tokenized_content = tokenizer(
                cleaned_content,
                max_length=100000,
                return_tensors='pt'
            ).to('cuda')
            document_length = tokenized_content['input_ids'].shape[1]
            num_chunks = math.ceil(document_length / CHUNK_LENGTH)

            for j in range(num_chunks):
                start = j*CHUNK_LENGTH
                end = (j+1)*CHUNK_LENGTH
                document_chunk = tokenized_content["input_ids"][0][start:end]
                decoded_content = tokenizer.decode(document_chunk)

                question = "In this paragraph, did the appellant claim any assignments of error or allegations about prosecutorial misconduct or misconduct by the state? If there is no information or mention about any allegations or assignments of error, answer no."

                full_prompt = (
                    f"Read the following paragraph from an appellate case carefully and think step-by-step through "
                    f"the following question, answering with only a 'Yes' or 'No'.  If you cannot determine the answer, provide your best yes or no guess: {question}\n\nParagraph from the case: {decoded_content}"
                )
                batch_messages.append([{"role": "user", "content": full_prompt}])

            batch_results = pipe(
                batch_messages,
                max_new_tokens=300,
                do_sample=False
            )


            label = "aoe_none"
            flipped = 0
            for i, result in enumerate(batch_results):
                if label_answers(result[0]["generated_text"][1]["content"]) == 1:
                    results.append({
                        "Index": batch.Index.iloc[counter],
                        "Response": result[0]["generated_text"][1]["content"],
                        "Text": decoded_content,
                        label: batch[label].iloc[counter]
                    })
                    flipped = 1
                    continue
            if not flipped:
                results.append({
                    "Index": batch.Index.iloc[counter],
                    "Response": result[0]["generated_text"][1]["content"],
                    label: batch[label].iloc[counter]
                })

            if batch_start % (batch_size * 5) == 0:
                gc.collect()
                torch.cuda.empty_cache()

            print(f"Processed up to sample {batch_end}")

            if batch_start % (batch_size * 10) == 0:
                temp_df = pd.DataFrame(results)
                temp_df.to_csv(f"./mistral_questions/{label}.csv.temp", index=False)
            
            counter += 1

    results_df = pd.DataFrame(results)

    results_df["Response Label"] = results_df["Response"].apply(label_flipped_answers)
    results_df.to_csv(f"./mistral_questions/{label}.csv", index=False)
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


if __name__ == "__main__":
    identify_allegations()