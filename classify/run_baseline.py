import os
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers import pipeline
import json
import gc

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def model_setup():
    model_name = "Equall/Saul-Instruct-v1"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        # device_map="auto",
    ).to('cuda')

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = 'right'
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_eos_token = True
    tokenizer.set_default_template = False

    return model, tokenizer


def clean_text(text):
    text = ' '.join(text.split())
    text = text.replace('\n', ' ')
    text = text.replace('\\t', ' ')
    text = text.replace('\\"', '"')
    return text



def run_generate(inference_df, filepath, model, tokenizer):
    for i in range(10):
        content = inference_df.Prompt.values[i] +  clean_text(inference_df.Context.values[i])
        formatted_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
        )
        input = tokenizer(formatted_prompt, max_length=20000, add_special_tokens=False, return_tensors='pt').to('cuda')
        decoded = tokenizer.decode(input["input_ids"][0], skip_special_tokens=True)
        response = model.generate(
            **input, 
            max_new_tokens=100, 
            do_sample=False,
            temperature=1.0,
            top_k=50,
            top_p=1.0
        )
        del input
        torch.cuda.empty_cache()
        decoded = tokenizer.decode(response[0], skip_special_tokens=True)

        del response
        torch.cuda.empty_cache()
        with open(filepath, "a") as f:
            f.write("\n\n\n<INSTRUCTION>\n\n\n")
            f.write(decoded)
        del decoded
        torch.cuda.empty_cache()

# Run model with pipeline
def run_pipeline(inference_df, filepath, model, tokenizer):

    pipe = pipeline("text-generation", model=model, torch_dtype=torch.bfloat16, device='cuda', tokenizer=tokenizer)
    pipe.model = pipe.model.to('cuda')


    for i in range(30):
        content = inference_df.Prompt.values[i] +  clean_text(inference_df.Context.values[i])
        input = tokenizer(content, max_length=20000, return_tensors='pt').to('cuda')
        decoded = tokenizer.decode(input["input_ids"][0][1:-1])
        message = [
            {"role": "user", "content": decoded},
            ]
        result = pipe(message, max_new_tokens=300, do_sample=False)
        with open(filepath, "a") as f:
            generated_text = result[0]["generated_text"]
            # f.write("\n\n\n<INSTRUCTION>\n\n\n")
            # f.write(generated_text[0]["content"])
            f.write("\n\n\n<RESPONSE>\n\n\n")
            f.write(generated_text[1]["content"])
            f.write(f"\n\n\n<LABEL>: {inference_df.Response.values[i]}")

# Run model with pipeline with closing reminder
def run_pipeline_with_closing_reminder(inference_df, filepath, model, tokenizer):

    pipe = pipeline("text-generation", model=model, torch_dtype=torch.bfloat16, device='cuda', tokenizer=tokenizer)
    pipe.model = pipe.model.to('cuda')


    for i in range(30):
        content = inference_df.Prompt.values[i] +  clean_text(inference_df.Context.values[i])
        tokenized_content = tokenizer(content, max_length=18000, return_tensors='pt').to('cuda')
        tokenized_content = tokenizer.decode(tokenized_content["input_ids"][0][1:-1])
        tokenized_content = tokenized_content + """\n\nAbove is the appellate case. Read over the case carefully and remember that your job is to determine whether the case meets some criteria for further evaluation of prosecutorial misconduct. """
        message = [
            {"role": "user", "content": tokenized_content},
            ]
        result = pipe(message, max_new_tokens=300, do_sample=False)
        with open(filepath, "a") as f:
            generated_text = result[0]["generated_text"]
            f.write("\n\n\n<RESPONSE>\n\n\n")
            f.write(generated_text[1]["content"])
            f.write(f"\n\n\n<LABEL>: {inference_df.Response.values[i]}")


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
    results_df.to_csv(filepath, index=False)


def questions_setup(): # Questions
    # case_juv_q = ("Is the defendant a juvenile (i.e. is the defendant younger than 18 years of age)? "
    #           "If the defendant's name is given as initials, if the appellant is referred to as a minor, "
    #           "or if the case is from juvenile court, the defendant is a juvenile. If no reference to the appellant "
    #               "being juvenile is made, the defendant is not a juvenile.")
    # case_crim_q = ("Is this case criminal? One indication that the case is criminal is if the trial case number"
    #            " contains the characters ‘CR’. If the trial case number does not contain 'CR,' another indication that "
    #                "the case might be criminal is if it is between the public and a private citizen. "
    #                "The case will not be criminal if the trial case number contains 'CV' for civil or 'CA' for civil "
    #                "appeal. Additionally, non criminal cases are between two private parties.")
    # case_2001_q = ("Did the original trial mentioned in this appellate case take place before 2001? If the original trial "
    #                "date is not mentioned, look for other clues that the trial might have taken place before 2001. The"
    #                " trial date will be before the conviction date and after the date of the crime.")
    # case_app_q = ("Is the appellee the city? If the appelle is listed as a city, not the state, the appellee is the city."
    #               "If the state or another party is listed as the appellee, the appelle is not the city.")
    # case_pros_q = "Is the prosecutor a city prosecutor?"
    # aoe_none_q = "Are there any allegations of prosecutorial misconduct mentioned?"
    # aoe_grandjury_q = "aoe_grandjury"
    aoe_court_q = "Is the allegation of error against the court, sometimes referred to as the “trial court”?"
    aoe_defense_q = "Is the allegation of error against the defense attorney?"
    aoe_procbar_q = ("Is the allegation procedurally barred? For example, is it barred by res judicata because it was not "
                 "raised during original trial and now it’s too late?")
    aoe_prochist_q = "aoe_prochist"
    # Question to variable mapping
    questions = {
        # case_juv_q: "case_juv",
        # case_crim_q: "case_crim",
        # case_2001_q: "case_2001",
        # case_app_q: "case_app",
        # case_pros_q: "case_pros",
        # aoe_none_q: "aoe_none",
        # aoe_grandjury_q: "aoe_grandjury",
        aoe_court_q: "aoe_court",
        aoe_defense_q: "aoe_defense",
        aoe_procbar_q: "aoe_procbar",
        aoe_prochist_q: "aoe_prochist",
    }
    return questions


def main():
    gc.collect()
    torch.cuda.empty_cache()
    model, tokenizer = model_setup()

    # path = "/home/amy_pu/pm-models/classify/inference.jsonl"
    # inference_df = pd.read_json(path, lines=True)
    # run_pipeline(inference_df, "pipeline_baseline_procedurally_barred.txt", model, tokenizer)
    # run_pipeline(inference_df, "outputs_baseline_truncated.txt", model, tokenizer)
    # run_pipeline_with_closing_reminder(inference_df, "outputs_baseline_closing_reminder.txt", model, tokenizer)
    # run_generate(inference_df, "generate_baseline.txt", model, tokenizer)

    questions = questions_setup()
    for q in questions:
        print(q)
        run_pipeline_with_questions(q, questions[q], f"{questions[q]}.csv", model, tokenizer)
        gc.collect()
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

if __name__ == "__main__":
   main()