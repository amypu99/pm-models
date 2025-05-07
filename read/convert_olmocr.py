import json

input_file_path = "../cases_olmocr/MS/ms_olmocr.jsonl"
output_file_path = "../cases_olmocr/MS/ms_olmocr-converted.jsonl"

# Open the file and process line by line
with open(input_file_path, 'r', encoding='utf-8') as input_file:
    with open(output_file_path, 'w') as output_file:
        for line in input_file:
            try:
                data = json.loads(line)
                context = data['text']
                filename = data['metadata']['Source-File']
                index = filename.replace('.pdf', '')
                index = index.replace('cases_pdf/MS/', '')
                index = index.replace('cases_temp/MS/', '')
                print(index)
                json_data = {}
                json_data['Index'] = index
                json_data['Context'] = context
                json_object = json.dumps(json_data, indent=4)
                output_file.write(json.dumps(json_object) + "\n")
            except json.JSONDecodeError as e:
                print(f"Error parsing line: {line}")
                print(e)
