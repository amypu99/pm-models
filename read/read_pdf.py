from pypdf import PdfReader
import os
import json
from random import shuffle
import pandas as pd

def process_to_txt(pdf_dir, output_dir, to_txt):
    """
    Creates txt files in output_dir for each pdf in pdf_dir
    """
    # Pdfs that have already been read into txt files
    processed_files = set(f.replace('.txt', '.pdf') for f in os.listdir(output_dir) if f.endswith('.txt'))
    # Iterate through pdfs not yet processed
    pdf_list = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf') and f not in processed_files]
    print(f"Found {len(pdf_list)} PDFs to process")
    for i, pdf in enumerate(pdf_list, start=1):
        print(f"Processing {i}, {pdf}")
        to_txt(pdf_dir, output_dir, pdf)

def to_txt(pdf_dir, output_dir, pdf):
    """
    Creates txt files in output_dir for pdf
    """
    filename = pdf.strip("pdf")
    output_path = output_dir + '/' + filename + 'txt'
    text = extract_text(pdf_dir, pdf)[0]
    with open(output_path, "w") as out_file:
        out_file.write(text)

def extract_text(pdf_dir, pdf):
    """
    Extracts text from pdf
    """
    pdf_path = pdf_dir + '/' + pdf
    reader = PdfReader(pdf_path)
    case_text = ""
    for page in reader.pages:
        case_text += page.extract_text()
    return case_text, reader.metadata

def make_jsonl(data_ms, data_dnms):
    """
    Creates jsonl file from data_ms and data_dnms
    """
    data = data_ms + data_dnms
    shuffle(data)
    train_len = int(len(data) * 0.8)
    test_len = len(data) - train_len
    train_data = data[:train_len]
    print(f"Train len: {len(train_data)}")
    test_data = data[train_len:]
    print(f"Test len: {len(test_data)}")
    with open("classify/train_data.jsonl", 'w') as file:
        for item in train_data:
            json_line = json.dumps(item)
            file.write(json_line + "\n")
    with open("classify/test_data.jsonl", 'w') as file:
        for item in test_data:
            json_line = json.dumps(item)
            file.write(json_line + "\n")

def process_to_jsonl(pdf_dir, to_jsonl):
    """
    Creates list of objects for each pdf in pdf_dir based on json object output of to_jsonl function
    ---json object format---
    json object format: {"Prompt": prompt, "Context": context, "Response": response}
    """
    pdf_list = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    print(f"Found {len(pdf_list)} PDFs to process")
    data = []
    prompt = ("I want to know whether or not this appellate case meets some criteria for further evaluation of "
              "prosecutorial misconduct. The case is in JSON format. Cases with any of the following do not meet the "
              "standards for further review: Allegation is procedurally barred (e.g. barred by res judicata because it "
              "was not raised during original trial and now it’s too late), Defendant is a juvenile, Case is not "
              "criminal, Allegation of error is against the court, not the prosecutor/state, Allegation is against "
              "defense attorney, not state, No allegation of prosecutorial misconduct, Misconduct is from a grand jury "
              "proceeding, Trial is before 2001, Allegation is in procedural history, Appellee is the city, not state, "
              "and Prosecutor is a city prosecutor, not county prosecutor. In your response, include whether the case "
              "meets standards or does not meet standards for future evaluation. If it does not meet standards for "
              "further evaluation, provide the reason why (from the above reasons listed). Please think through this "
              "task step-by-step. Use the following text from the case to make your decision:")
    for i, pdf in enumerate(pdf_list, start=1):
        print(f"Processing {i}, {pdf}")
        data.append(to_jsonl(prompt, pdf_dir, pdf))
    return data

def to_jsonl_ms(prompt, pdf_dir, pdf):
    """
    Creates json object for pdfs that meet standards
    ---json object format---
    {"Prompt": prompt, "Context": context, "Response": 'Meets Standards'}
    """
    context = extract_text(pdf_dir, pdf)[0]
    response = "Meets standards"
    return {"Prompt": prompt, "Context": context, "Response": response}

def to_jsonl_dnms(prompt, pdf_dir, pdf):
    """
    Creates json object for pdfs that meet standards
    ---json object format---
    {"Prompt": prompt, "Context": context, "Response": 'Does not meet standards because [reasons]'}
    """
    df = pd.read_csv('Cases-With-Metadata.csv')
    metadata = extract_text(pdf_dir, pdf)[1]
    matching = df[df['Filename'] == pdf]
    reasons = ""
    if not matching.empty:
        row = df.loc[df['Filename'] == pdf]
        reasons = str(row['Standards Not Met'].values[0])
        reasons = reasons.replace("\"", " ")
    if matching.empty:
        print("No matches found")
    context = extract_text(pdf_dir, pdf)[0]
    response = "Does not meet standards because " + reasons
    return {"Prompt": prompt, "Context": context, "Response": response}


def main():
    dnms_dir = "cases_pdf/DNMS_2018to2020"
    ms_dir = "cases_pdf/MS_2018to2020"
    data_dnms = process_to_jsonl(dnms_dir, to_jsonl_dnms)
    data_ms = process_to_jsonl(ms_dir, to_jsonl_ms)
    make_jsonl(data_ms, data_dnms)


if __name__ == "__main__":
    main()



