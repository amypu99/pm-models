from get_all_aoe import full_query
import os
import json
import re
import gc
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline
)
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from run_baseline import clean_text, mistral_setup, ministral_setup, llama_setup
from run_questions import find_whole_word, load_jsonl



os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# with open('./list_of_allegations.jsonl', 'r') as json_file:
#     json_list = list(json_file)
#
# for json_str in json_list:
#     result = json.loads(json_str)
#     allegations = result['allegations']
#     text = re.sub(r'^```json\s*|\s*```$', '', allegations.strip())
#     print(text)
#     allegation_list = json.loads(text)
#     print(allegation_list)


def iterate_over_allegations(model, tokenizer):
    full_query(model, tokenizer)


if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    filepath = "../cases_olmocr/MS/ms_olmocr_converted.jsonl"
    all_jsonl = load_jsonl(filepath)
    all_jsonl = all_jsonl.sample(n=5, random_state=42)
    results = {}
    temp_results = {}

    model, tokenizer = ministral_setup()
    pipe = pipeline("text-generation", model=model, max_new_tokens=256, torch_dtype=torch.bfloat16,
                    device_map='cuda', tokenizer=tokenizer)
    pipe.model = pipe.model.to('cuda')


    for index, row in all_jsonl.iterrows():
        if "Context" in row:
            print(str(row["Index"]))
            context = str(row["Context"])
            allegation_list_reg = full_query(model, tokenizer, context, results)

            # Error checking allegation list
            try:
                allegation_list = json.loads(allegation_list_reg)
                print("starting iteration")
                print(allegation_list)
            except json.decoder.JSONDecodeError:
                max_attempts = 5
                attempts = 0
                success = False

                while not success and attempts < max_attempts:
                    print("redo")
                    attempts += 1
                    allegation_list_reg = full_query(model, tokenizer)
                    try:
                        allegation_list = json.loads(allegation_list_reg)
                        success = True
                        print(allegation_list)
                    except json.decoder.JSONDecodeError:
                        if attempts >= max_attempts:
                            print(f"Failed after 5 attempts. Giving up.")

            # Marker for meeting standards
            meets_standards = False
            for attribute, value in allegation_list.items():
                # Ask if aoe is against the prosecutor
                print(attribute, value)
                q = "Is the assignment of error alleging that the prosecutor or State committed misconduct?"
                question = (
                    f"{value}\n\n"
                    "Above is an assignment of error from an appellate case. Read over the assignment of error carefully and think step-by-step through "
                    f"the following question, answering with only a 'Yes' or 'No'.  If you cannot determine the answer, provide your best yes or no guess: {q}"
                )
                prompt_aoe_none = [{"role": "user", "content": question}]
                results_aoe_none = pipe(prompt_aoe_none, max_new_tokens=256)

                # If aoe is against prosecutor
                if find_whole_word("Yes")(results_aoe_none[2]['content']):
                    # Ask if aoe is procedurally barred
                    q = "Is the assignment of error procedurally barred?"
                    prompt_aoe_procbar = [{"role": "user", "content": question}]
                    results_aoe_procbar = pipe(prompt_aoe_procbar, max_new_tokens=256)

                    # If aoe is not procedurally barred
                    if find_whole_word("No")(results_aoe_procbar[2]['content']):
                        # Ask if aoe is in the procedural history
                        q = "Is the assignment of error in the procedural history"
                        prompt_aoe_prochist = [{"role": "user", "content": question}]
                        results_aoe_prochist = pipe(prompt_aoe_prochist, max_new_tokens=256)

                        # If aoe is not in procedural history
                        if find_whole_word("No")(results_aoe_prochist[2]['content']):
                            # HAS TO REACH HERE TO MEET STANDARDS FOR AT LEAST ONE ALLEGATION
                            meets_standards = True

                    # If aoe is procedurally barred
                    else:
                        continue

                # If aoe is not against prosecutor, continue
                else:
                    continue