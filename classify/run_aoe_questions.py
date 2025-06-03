import pandas as pd
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
from run_case_questions import find_whole_word, load_jsonl
from run_pipeline import filter_jsonl


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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


def full_logic():
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
    # pipe.model = pipe.model.to('cuda')


    # For each case
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

def condensed_logic():
    ms_index_list = load_jsonl("../cases_olmocr/MS/ms_olmocr_converted_with_label.jsonl")["Index"].tolist()

    gc.collect()
    torch.cuda.empty_cache()
    filepath = "./results/extracted_join.jsonl"
    output_path = "./results/aoe_test/aoe_questions_results_join_20250603"
    # full_jsonl = load_jsonl(filepath)
    # aoe_procbar2_df = pd.read_csv("./results/pipeline_test_2025-05-26/aoe_procbar2.csv")
    # aoe_evidence_jsonl = filter_jsonl(aoe_procbar2_df, full_jsonl)
    aoe_evidence_jsonl = load_jsonl(filepath)
    # aoe_evidence_jsonl = aoe_evidence_jsonl.sample(n=2, random_state=42)
    results = {}
    temp_results = {}

    model, tokenizer = ministral_setup()
    pipe = pipeline("text-generation", model=model, max_new_tokens=256, torch_dtype=torch.bfloat16,
                    device_map='cuda', tokenizer=tokenizer)
    # pipe.model = pipe.model.to('cuda')

    # For each case
    question_results = []
    case_results = []
    for case_name, group in aoe_evidence_jsonl.groupby('index'):

        gold_label = 0 if case_name in ms_index_list else 1
        meets_standards = False
        explanation = ""
        print("gold_label:", gold_label)

        for index, row in group.iterrows():
            if "allegation" in row:
                    case_name = str(row["index"])
                    allegation_num = str(row["allegation_num"])
                    aoe = str(row["allegation"])
                    aoe_evidence = str(row["extracted_text"])

                    print(case_name, aoe)

                    # Ask if aoe is against the prosecutor, is procedurally barred, or is in the procedural history
                    question = (f"{aoe}:{aoe_evidence}\n\n" "Above is an assignment of error from an appellate case and "
                                "its explanation given in the format 'assignment of error':'extracted_text'. Read over the "
                                "both carefully and think step-by-step through the following questions, answering with "
                                "only a 'Yes' or 'No.' If you cannot determine the answer, provide your best yes or "
                                "no guess:\n\n"
                                "### Questions \n"
                                "1a. Is the prosecutor (also called prosecution) involved at all?\n"
                                "1b. Is the state involved at all?\n"
                                "1c. Is prosecutorial misconduct mentioned at all?\n"
                                "2. Is the assignment of error procedurally barred by res judicata?\n"
                                "3. Is the assignment of error in the procedural history (i.e. is it from a past appeal))\n"                            
                                "### Output format:\n\n"
                                "Return only this JSON block and nothing else:"
                                "```json{\"aoe_none_1b\": \"<answer to question 1a>\",\"aoe_none_1b\": \"<answer to question "
                                "1b>\", \"aoe_none_1c\": \"<answer to question 1c>\", \"aoe_procbar\": \"<answer to "
                                "question 2>\", \"aoe_prochist\": \"<answer to question 3>\"}```")

                    prompt_all = [{"role": "user", "content": question}]
                    results = pipe(prompt_all, max_new_tokens=256)[0]['generated_text']

                    aoe_answers = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', results[1]['content'], re.DOTALL).group(1)

                    try:
                        aoe_answers = json.loads(aoe_answers)
                    except json.decoder.JSONDecodeError:
                        print(f"Skipping {case_name}, {index}. Not able to load as json object")
                        continue

                    if (find_whole_word("Yes")(aoe_answers['aoe_none_1a']) or find_whole_word("Yes")(aoe_answers['aoe_none_1b'])
                            or find_whole_word("Yes")(aoe_answers['aoe_none_1c'])):
                        print("Allegation is against Prosecutor")
                        if find_whole_word("No")(aoe_answers['aoe_procbar']):
                            print("Allegation is not procedurally barred")
                            if find_whole_word("No")(aoe_answers['aoe_prochist']):
                                print("Allegation is not in procedural history")
                                meets_standards = True
                                if meets_standards:
                                    print("CASE MEETS STANDARDS")
                                    question_results.append({
                                        "index": case_name,
                                        "allegation_num": allegation_num,
                                        "allegation": aoe,
                                        "result": 0,
                                        "explanation": "meets standards"
                                    })
                            else:
                                print("Allegation is in procedural history")
                                explanation = "procedural history"
                                question_results.append({
                                    "index": case_name,
                                    "allegation_num": allegation_num,
                                    "allegation": aoe,
                                    "result": 1,
                                    "explanation": explanation
                                })
                        else:
                            print("Allegation is procedurally barred")
                            explanation = "procedurally barred"
                            question_results.append({
                                "index": case_name,
                                "allegation_num": allegation_num,
                                "allegation": aoe,
                                "result": 1,
                                "explanation": explanation
                            })
                    else:
                        print("Allegation is not against Prosecutor")
                        explanation = "not prosecutor"
                        question_results.append({
                            "index": case_name,
                            "allegation_num": allegation_num,
                            "allegation": aoe,
                            "result": 1,
                            "explanation": explanation
                        })
        if meets_standards:
            case_results.append({
                "Index": case_name,
                "Gold Label": gold_label,
                "Predicted Label": 0,
            })
        else:
            case_results.append({
                "Index": case_name,
                "Gold Label": gold_label,
                "Predicted Label": 1,
                "Explanation": explanation,
            })


    aoe_results_df = pd.DataFrame(question_results)
    aoe_results_df.to_csv(output_path + ".csv", index=False)
    collapsed_results_df = pd.DataFrame(case_results)
    collapsed_results_df.to_csv(output_path + "_collapsed.csv", index=False)


if __name__ == "__main__":
    condensed_logic()