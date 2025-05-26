import pandas as pd
import torch
from transformers import pipeline
import json
import gc
import re
from run_baseline import clean_text, mistral_setup, ministral_setup
from datetime import date


def load_jsonl(filepath):
    data = []
    with open(filepath, 'r') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return pd.DataFrame(data)


def run_question(question=None, cases_jsonl=None, prompt_func=None, label=None, label_func=None, model=None, tokenizer=None, batch_size=4):
    label1 = label
    if label=="aoe_procbar1" or label=="aoe_procbar2":
        label1 = "aoe_procbar"

    pipe = pipeline(
        "text-generation",
        model=model,
        torch_dtype=torch.bfloat16,
        tokenizer=tokenizer
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
            temp_df["Predicted Label"] = temp_df["Response"].apply(label_func)
            temp_df.to_csv(f"./results/pipeline_test_{date.today()}/{label}.csv.temp", index=False)

    results_df = pd.DataFrame(results)
    results_df["Predicted Label"] = results_df["Response"].apply(label_func)
    results_df.to_csv(f"./results/pipeline_test_{date.today()}/{label}.csv", index=False)

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
            "Gold Label": batch[label].iloc[i],
            "Response": result[0]["generated_text"][1]["content"]
        })


def run_ordered_question_pipeline(question, label, q_df, model, tokenizer, batch_size=4):
    dnms_jsonl = load_jsonl("jsonl/dnms.jsonl")
    ms_jsonl = load_jsonl("jsonl/ms.jsonl")

    pipe = pipeline(
        "text-generation",
        model=model,
        torch_dtype=torch.bfloat16,
        tokenizer=tokenizer
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
            # Match index in either JSONL file
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
                    return_tensors='pt'
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
                "Gold Label": matched_row[label].iloc[0],  # Get label from the matched row
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


# def questions_setup():
    # Question to variable mapping
    # questions = {
        # "case_juv": "Is the defendant a juvenile (i.e. is the defendant younger than 18 years of age)? Some hints that the"
        #           " defendant is not juvenile are if the defendant's name is given as initials, if the appellant is "
        #           "referred to as minor, or if the case is from juvenile court. If you cannot determine the answer or no"
        #           " reference to the defendant being juvenile is made, the defendant is not a juvenile.",
        # "case_crim": "Is the case criminal? Criminal cases are between the public and a private citizen. Civil cases are "
        #            "between two private parties. (Hint: Case is criminal if the trial case number contains the "
        #            "characters ‘CR’. Case is civil if the trial case number contains 'CV' or 'CA', or if the case is "
        #            "marked as a civil appeal).",
        # "case_2001": "Did the original trial mentioned in this appellate case take place before 2001? The"
        #            " trial date will be before the conviction date and after the date of the crime.",
        # "case_app": "Is the appellee the city? If the appelle is listed as a city, not the state, the appellee is the city."
        #           "If the state or another party is listed as the appellee, the appelle is not the city.",
        # "case_pros" : "Is the prosecutor in question a city prosecutor?",
        # "aoe_none": "Identify and summarize each assignment of error. Do any of them mention prosecutorial misconduct/error or misconduct/error by the state?",
        # aoe_grandjury_q: "aoe_grandjury",
        # "aoe_court": "Is the allegation of error against the court, sometimes referred to as the 'trial court'?",
        # "aoe_defense": "Is the allegation of error against the defense attorney?",
        # "aoe_procbar": "Is the allegation procedurally barred? For example, is it barred by res judicata because it was not "
        #          "raised during original trial and now it’s too late?",
        # "aoe_prochist": "Is the allegation in procedural history, i.e., was the prosecutorial misconduct in question raised"
        #               " in a previous appeal?"
    # }
    # return questions

def questions_setup():
    # Question to variable mapping
    questions = {
        "aoe_procbar1": "Does the text above indicate that the assignments of error were procedurally barred because the"
                        " appellant filed an untimely appeal? Answer with only a 'Yes' or 'No'.  If you cannot determine"
                        " the answer, provide your best yes or no guess.",
        "aoe_procbar2": "Does the text above indicate that the assignments of error were procedurally barred because the "
                        "appellant failed to properly file for appeal? Answer with only a 'Yes' or 'No'.  If you cannot "
                        "determine the answer, provide your best yes or no guess.",
        "case_2001": "What is the year of the original trial case? Look for the year of the crime or the year of "
                     "conviction or the year of indictment. The trial year will be before the conviction year. The trial "
                     "year will be after the year of the crime and also before the year of indictment. Do NOT respond "
                     "with the date of a citation."
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
    dnms_jsonl = load_jsonl("jsonl/dnms.jsonl")
    ms_jsonl = load_jsonl("jsonl/ms.jsonl")
    dnms_sample = dnms_jsonl.sample(n=100, random_state=42)
    ms_sample = ms_jsonl.sample(n=100, random_state=42)
    question_jsonl = pd.concat([dnms_sample, ms_sample]).reset_index(drop=True)

    gc.collect()
    torch.cuda.empty_cache()
    model, tokenizer = ministral_setup()
    question_dict = questions_setup()
    for q in question_dict:
        print(question_dict[q])
        results_df = run_question(question_dict[q], question_jsonl, q, model, tokenizer)
        results_df["Predicted Label"] = results_df["Response"].apply(label_answers)
        results_df.to_csv(f"./standards_csv/{q}_filtered.csv", index=False)
        gc.collect()
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()


def flip_binary(df):
    df["Predicted Label"] = df["Predicted Label"].apply(lambda x: 0 if x == 1 else 1)
    return df


def run_ordered():
    gc.collect()
    torch.cuda.empty_cache()
    model, tokenizer = mistral_setup()

    eval_df = pd.read_csv("standards_csv/evaluation_results.csv")
    eval_df = eval_df.sort_values(by=["f1_score"], ascending=False)
    questions = eval_df["filename"].tolist()
    questions.remove("aoe_defense")
    questions.remove("aoe_court")

    dnms_df = pd.read_csv("standards_csv/case_app_regex.csv")
    ms_df = pd.read_csv("standards_csv/ms_case_app_regex.csv")
    q_df = pd.concat([dnms_df, ms_df], axis=0, ignore_index=True)
    q_df = q_df[q_df['Predicted Label'] == 0]

    questions_dict = questions_setup()
    results_df = pd.DataFrame()

    for q in questions:
        print(f"\nRunning question: {questions_dict[q]}")
        results_df = run_ordered_question_pipeline(questions_dict[q], q, q_df, model, tokenizer)

        # Update response label
        results_df["Predicted Label"] = results_df["Response"].apply(label_answers)


        if q=="case_crim" or q=="aoe_none":
            results_df = flip_binary(results_df)
            print("flipped")

        results_df.to_csv(f"./mistral_run/{q}.csv", index=False)

        # Filter rows where Response Label == 0 for the next iteration
        q_df = results_df[results_df['Predicted Label'] == 0]
        new_q_df_size = len(results_df[results_df['Predicted Label'] == 0])
        print(f"New q_df size after filtering: {new_q_df_size}")

        # Free up GPU memory
        gc.collect()
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    results_df.to_csv(f"./mistral_run/final.csv", index=False)


if __name__ == "__main__":
    run_specified()

