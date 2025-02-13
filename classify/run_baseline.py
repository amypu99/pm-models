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
run_pipeline_with_closing_reminder("outputs_baseline_closing_reminder.txt")
# run_generate("generate_baseline.txt")

