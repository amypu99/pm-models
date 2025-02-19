import os
import pandas as pd
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
import torch
from transformers import pipeline
import torch

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


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


path = "/home/amy_pu/pm-models/classify/inference.jsonl"
inference_df = pd.read_json(path, lines=True)

def clean_text(text):
    text = ' '.join(text.split())

    text = text.replace('\n', ' ')
    text = text.replace('\\t', ' ')
    text = text.replace('\\"', '"')

    return text

torch.cuda.empty_cache()

def run_generate(filepath):
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
def run_pipeline(filepath):

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
def run_pipeline_with_closing_reminder(filepath):

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

# run_pipeline("pipeline_baseline_procedurally_barred.txt")
# run_pipeline("outputs_baseline_truncated.txt")
# run_pipeline_with_closing_reminder("outputs_baseline_closing_reminder.txt")
# run_generate("generate_baseline.txt")


dnms_path = "dnms.jsonl"
question_df = pd.read_json(dnms_path, lines=True)

# Questions
case_juv_q = ("Is the defendant a juvenile (i.e. is the defendant younger than 18 years of age)? "
              "If the defendant's name is given as initials or if the appellant is referred to as a minor, "
              "the defendant is a juvenile. ")
case_crim_q = ("Is the case criminal? One indicator that the case is criminal is if the trial case number"
               " includes the characters ‘CR’.")
# case_2001_q = "case_2001"
case_app_q = "Is the appellee the city? "
case_pros_q = "Is the prosecutor a city prosecutor?"
aoe_none_q = "Are there any allegations of prosecutorial misconduct?"
# aoe_grandjury_q = "aoe_grandjury"
aoe_court_q = "Is the allegation of error against the court, sometimes referred to as the “trial court”?"
aoe_defense_q = "Is the allegation of error against the defense attorney?"
aoe_procbar_q = ("Is the allegation procedurally barred? For example, is it barred by res judicata because it was not "
                 "raised during original trial and now it’s too late?")
# aoe_prochist_q = "aoe_prochist"


def run_pipeline_with_questions(question, label, filepath):

    pipe = pipeline("text-generation", model=model, torch_dtype=torch.bfloat16, device='cuda', tokenizer=tokenizer)
    pipe.model = pipe.model.to('cuda')

    results = []

    for i in range(30):
        content = question_df.Prompt.values[i] + clean_text(question_df.Context.values[i])
        tokenized_content = tokenizer(content, max_length=18000, return_tensors='pt').to('cuda')
        tokenized_content = tokenizer.decode(tokenized_content["input_ids"][0][1:-1])
        tokenized_content = tokenized_content + """\n\nAbove is the appellate case. Read over the case carefully and 
        answer the following question: """ + question
        message = [
            {"role": "user", "content": tokenized_content},
        ]

        result = pipe(message, max_new_tokens=300, do_sample=False)

        generated_text = result[0]["generated_text"][1]["content"]
        results.append({
            "Index": question_df.Index.iloc[i],
            "Response": generated_text,
            "Label": question_df.label.values[i],
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(filepath, index=False)


# Question to variable mapping
questions = {
        case_juv_q: "case_juv",
        case_crim_q: "case_crim",
        # case_2001_q: "case_2001",
        case_app_q: "case_app",
        case_pros_q: "case_pros",
        aoe_none_q: "aoe_none",
        # aoe_grandjury_q: "aoe_grandjury",
        aoe_court_q: "aoe_court",
        aoe_defense_q: "aoe_defense",
        aoe_procbar_q: "aoe_procbar",
        # aoe_prochist_q: "aoe_prochist",
    }

for q in questions:
    run_pipeline_with_questions(q, questions[q], f"{questions[q]}.csv")
