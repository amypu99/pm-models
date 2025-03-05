import os
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers import pipeline
import json
import gc
import re
from run_baseline import clean_text, model_setup


def load_jsonl(filepath):
    data = []
    with open(filepath, 'r') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return pd.DataFrame(data)


def run_pipeline_with_questions(question, label, filepath, model, tokenizer, batch_size=4):
    question_df = load_jsonl("dnms.jsonl")

    pipe = pipeline(
        "text-generation",
        model=model,
        torch_dtype=torch.bfloat16,
        device='cuda',
        tokenizer=tokenizer
    )
    pipe.model = pipe.model.to('cuda')

    results = []

    for batch_start in range(0, 100, batch_size):
        batch_end = min(batch_start + batch_size, len(question_df))
        batch = question_df.iloc[batch_start:batch_end]

        batch_messages = []
        for content in batch.Context.values:
            cleaned_content = clean_text(content)
            tokenized_content = tokenizer(
                cleaned_content,
                max_length=20000,
                return_tensors='pt'
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
                "Response": result[0]["generated_text"][1]["content"],
                label: batch[label].iloc[i]
            })

        if batch_start % (batch_size * 5) == 0:
            gc.collect()
            torch.cuda.empty_cache()

        print(f"Processed up to sample {batch_end}")

        if batch_start % (batch_size * 10) == 0:
            temp_df = pd.DataFrame(results)
            temp_df.to_csv(f"{filepath}.temp", index=False)

    results_df = pd.DataFrame(results)
    return results_df


def questions_setup(): # Questions
    # case_juv_q = ("Is the defendant a juvenile (i.e. is the defendant younger than 18 years of age)? Some hints that the"
    #               " defendant is not juvenile are if the defendant's name is given as initials, if the appellant is "
    #               "referred to as minor, or if the case is from juvenile court. If you cannot determine the answer or no"
    #               " reference to the defendant being juvenile is made, the defendant is not a juvenile.")
    # case_crim_q = ("Is the case criminal? Criminal cases are between the public and a private citizen. Civil cases are "
    #                "between two private parties. (Hint: Case is criminal if the trial case number contains the "
    #                "characters ‘CR’. Case is civil if the trial case number contains 'CV' or 'CA', or if the case is "
    #                "marked as a civil appeal).")
    # case_2001_q = ("Did the original trial mentioned in this appellate case take place before 2001? If the original trial "
    #                "date is not mentioned, look for other clues that the trial might have taken place before 2001. The"
    #                " trial date will be before the conviction date and after the date of the crime.")
    # case_app_q = ("Is the appellee the city? If the appelle is listed as a city, not the state, the appellee is the city."
    #               "If the state or another party is listed as the appellee, the appelle is not the city.")
    # case_pros_q = "Is the prosecutor a city prosecutor?"
    # aoe_none_q = "Are there any allegations of prosecutorial misconduct mentioned?"
    # aoe_grandjury_q = "aoe_grandjury"
    # aoe_court_q = "Is the allegation of error against the court, sometimes referred to as the “trial court”?"
    # aoe_defense_q = "Is the allegation of error against the defense attorney?"
    # aoe_procbar_q = ("Is the allegation procedurally barred? For example, is it barred by res judicata because it was not "
    #              "raised during original trial and now it’s too late?")
    aoe_prochist_q = ("Is the allegation in procedural history, i.e., was the prosecutorial misconduct in question raised"
                      " in a previous appeal?")
    # Question to variable mapping
    questions = {
        # case_juv_q: "case_juv",
        # case_crim_q: "case_crim",
        # case_2001_q: "case_2001",
        # case_app_q: "case_app",
        # case_pros_q: "case_pros",
        # aoe_none_q: "aoe_none",
        # aoe_grandjury_q: "aoe_grandjury",
        # aoe_court_q: "aoe_court",
        # aoe_defense_q: "aoe_defense",
        # aoe_procbar_q: "aoe_procbar",
        aoe_prochist_q: "aoe_prochist"
    }
    return questions

def find_whole_word(w):

    return re.compile(r'\b({0})\b'.format(w), flags=re.IGNORECASE).search

def label_answers(answer):
    answer = answer.replace(".", "")
    if find_whole_word("Yes")(answer) or find_whole_word("Answer: Yes")(answer):
            return 1
    elif find_whole_word("No")(answer) or find_whole_word("Answer: No")(answer) or find_whole_word("the appellee is not the city")(answer):
        return 0
    else:
        return 99

def label_flipped_answers(answer):
    answer.replace(".", "")
    if find_whole_word("Yes")(answer) or find_whole_word("Answer: Yes")(answer):
            return 0
    if find_whole_word("No")(answer) or find_whole_word("Answer: No")(answer):
        return 1
    else:
        return 99


def run_specified():
    gc.collect()
    torch.cuda.empty_cache()
    model, tokenizer = model_setup()

    questions = questions_setup()
    for q in questions:
        print(q)
        results_df = run_pipeline_with_questions(q, questions[q], f"./standards_csv/{questions[q]}.csv", model, tokenizer)
        results_df["Response Label"] = results_df["Response"].apply(label_answers)
        results_df.to_csv(f"./standards_csv/{questions[q]}.csv", index=False)
        gc.collect()
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()



if __name__ == "__main__":
    run_specified()