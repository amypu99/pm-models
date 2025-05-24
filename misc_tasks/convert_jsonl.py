 import json

def create_indices():
    # Input and output file paths
    input_file = '../cases_olmocr/DNMS/dnms_olmocr.jsonl'
    output_file = '../cases_olmocr/DNMS/dnms_olmocr_temp.jsonl'

    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            data = json.loads(line)
            index = data['metadata']['Source-File'].replace('.pdf', '').replace("cases_pdf/MS/", "").replace("cases_pdf/DNMS/", "")
            context = data['text']
            
            new_data = {
                'Index': index,
                'Context': context
            }
            outfile.write(json.dumps(new_data) + '\n')

    print(f"Converted file saved to {output_file}")

def get_labels():

    # File paths
    file1 = '../cases_olmocr/DNMS/dnms_olmocr_temp.jsonl'  # The file with "Index" and "Context"
    file2 = './jsonl/dnms.jsonl'  # The file with "Index", "Context", and metadata
    output_file = '../cases_olmocr/DNMS/dnms_olmocr_converted.jsonl'

    # Load second file into a dictionary keyed by Index
    metadata_map = {}
    with open(file2, 'r', encoding='utf-8') as f2:
        for line in f2:
            data = json.loads(line)
            index = data['Index']
            # Exclude Context from second file, keep the rest
            metadata = {k: v for k, v in data.items() if k not in ('Index', 'Context')}
            metadata_map[index] = metadata

    # Process first file and merge with metadata
    with open(file1, 'r', encoding='utf-8') as f1, \
        open(output_file, 'w', encoding='utf-8') as out:
        
        for line in f1:
            data = json.loads(line)
            index = data['Index']
            context = data['Context']
            
            merged = {
                'Index': index,
                'Context': context,
            }
            
            # Merge metadata from file2 if exists
            if index in metadata_map:
                merged.update(metadata_map[index])
        
            out.write(json.dumps(merged) + '\n')

    print(f"Merged file written to {output_file}")


get_labels()
# create_indices()