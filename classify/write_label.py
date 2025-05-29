import json

input_file = '../cases_olmocr/DNMS/dnms_olmocr_converted.jsonl'   # Replace with your input file name
output_file = '../cases_olmocr/DNMS/dnms_olmocr_converted_with_label.jsonl' # Desired output file name

with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    for line in infile:
        record = json.loads(line)
        record['meets_standards'] = 0
        outfile.write(json.dumps(record) + '\n')