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
from run_questions import label_flipped_answers, label_answers, load_jsonl



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
    for field in all_jsonl:
        if field == "Context":
            context = str(all_jsonl[field])
            allegation_list = full_query(model, tokenizer, context, results)
            print(allegation_list)
            # try:
            #     # allegation_list = json.loads(allegation_list)
            #     print(allegation_list)
            # except json.decoder.JSONDecodeError:
            #     allegation_list = full_query(model, tokenizer)
            #     print(allegation_list)
