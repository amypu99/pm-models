import os
import gc
import torch
import json
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
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

if __name__ == "__main__":
    # from huggingface_hub import login
    # login()
    gc.collect()
    torch.cuda.empty_cache()
    filepath = "./list_of_allegations_20250520.jsonl"
    all_jsonl = load_jsonl(filepath)
    print(all_jsonl)
    # results = {}
    # temp_results = {}

    import json

    # input_file = 'input.jsonl'   # Change this to your input file path
    output_file = './list_of_allegations_20250520_2.jsonl' # Change this to your desired output file path

    prefix_to_remove = "cases_temp/MS/"

    with open(filepath, 'r', encoding='utf-8') as infile, \
        open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            record = json.loads(line)
            
            # Modify index if it starts with the prefix
            if record.get('index', '').startswith(prefix_to_remove):
                record['index'] = record['index'][len(prefix_to_remove):]
            
            # Write the modified (or unmodified) record to the output file
            outfile.write(json.dumps(record) + '\n')

        print(f"Finished processing. Cleaned file saved as '{output_file}'.")

    # model, tokenizer = ministral_setup()
    # pipe = pipeline("text-generation", model=model, max_new_tokens=1200, torch_dtype=torch.bfloat16, device_map='cuda', tokenizer=tokenizer)
    # pipe.model = pipe.model.to('cuda')

    # for i, row in all_jsonl.iterrows():
    #     key = row['index']
    #     allegations = row['allegations']
    #     allegations_obj = allegations.replace("```", "").replace("json", "")
    #     if key == "253-FORTUNE II" or key == "420-Quinn":
    #         continue
    #     print(key)
    #     print(json.loads(allegations_obj))
        # allegation_list = full_query(pipe, tokenizer, row["Context"])
        # with open("list_of_allegations_20250220.jsonl", "a") as f:
        #     json_record = json.dumps({"index": key, "allegations": allegation_list})
        #     f.write(json_record + "\n")
        # gc.collect()
        # torch.cuda.empty_cache()