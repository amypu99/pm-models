import torch
from transformers import pipeline
import os


with open("classify_prompt.txt", "r") as file:
    prompt_content = file.read().strip()

pipe = pipeline("text-generation", model="Equall/Saul-Instruct-v1", torch_dtype=torch.bfloat16, device=0)

# output_file = "MS_check_outputs.txt"
# with open(output_file, "w") as out_file:
#     for f in os.listdir("MS_check"):
#         try:
#             with open(f"MS_check/{f}", "r", encoding="utf-8") as file:
#                 case_content = file.read().strip()
#         except UnicodeDecodeError:
#             with open(f"MS_check1/{f}", "r", encoding="latin-1") as file:
#                 case_content = file.read().strip()
#
#         final_prompt = prompt_content + case_content
#
#         messages = [{"role": "user", "content": final_prompt}]
#         prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#         outputs = pipe(prompt, max_new_tokens=256, do_sample=False, return_full_text=False)
#
#         out_file.write(f"File: {f}\n")
#         out_file.write(f"Model Output:\n{outputs[0]['generated_text']}\n\n")
#
# print(f"All outputs written to {output_file}.")

messages = [{"role": "user", "content": "hello world"}]
prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
outputs = pipe(prompt, max_new_tokens=256, do_sample=False, return_full_text=False)
print(outputs[0]['generated_text'])