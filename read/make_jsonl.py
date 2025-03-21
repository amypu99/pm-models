"""
**Creating Jsonl**
This script has functions to create various json files for prompting, including
    * Train/Test jsonl
    * Inference jsonl
    * Question jsonl
Assumes downloaded: Cases-With-Metadata.csv, cases_pdf/MS_2018_2021, cases_pdf/DNMS_2018_2021,
cases_pdf/DNMS_2021, cases_pdf/MS_2021
Dependencies: os, json, read_pdf, pandas, random
"""
import os
import json
import read_pdf
import pandas as pd
from random import shuffle

def make_jsonl(data_ms, data_dnms):
    """
    Creates train and test jsonl files
    Arguments: output of process_to_jsonl
    """
    data = data_ms + data_dnms
    shuffle(data)
    # Set train and test sizes
    train_len = int(len(data) * 0.8)
    train_data = data[:train_len]
    print(f"Train len: {len(train_data)}")
    test_data = data[train_len:]
    print(f"Test len: {len(test_data)}")
    # Write json objects into separate jsonl files
    with open("classify/train_data.jsonl", 'w') as file:
        for item in train_data:
            json_line = json.dumps(item)
            file.write(json_line + "\n")
    with open("classify/test_data.jsonl", 'w') as file:
        for item in test_data:
            json_line = json.dumps(item)
            file.write(json_line + "\n")

def make_jsonl_inference(data_ms, data_dnms):
    """
    Creates jsonl file for inference
    """
    data = data_ms + data_dnms
    with open("./inference.jsonl", 'w') as file:
        for item in data:
            json_line = json.dumps(item)
            file.write(json_line + "\n")

def make_jsonl_questions(pdf_dir, filename, to_json):
    """
    Creates jsonl file for questions
    """
    pdf_list = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    print(f"Found {len(pdf_list)} PDFs to process")
    data = []
    # For every PDF in the directory of case PDFs, make a json object
    for i, pdf in enumerate(pdf_list, start=1):
        print(f"Processing {i}, {pdf}")
        data.append(to_json(pdf_dir, pdf))

    # Write json objects to jsonl file
    with open(filename, 'w') as file:
        for item in data:
            json_line = json.dumps(item)
            file.write(json_line + "\n")


def process_to_jsonl(pdf_dir, to_jsonl):
    """
    Creates list of json objects for each pdf in pdf_dir based on json object output of to_jsonl function
    """
    pdf_list = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    print(f"Found {len(pdf_list)} PDFs to process")
    data = []
    prompt = ("I want to know whether or not this appellate case meets some criteria for further evaluation of "
              "prosecutorial misconduct. Cases with any of the following do not meet the "
              "standards for further review: Allegation is procedurally barred (e.g. barred by res judicata because it "
              "was not raised during original trial and now it’s too late), Defendant is a juvenile, Case is not "
              "criminal, Allegation of error is against the court, not the prosecutor/state, Allegation is against "
              "defense attorney, not state, No allegation of prosecutorial misconduct, Misconduct is from a grand jury "
              "proceeding, Trial is before 2001, Allegation is in procedural history, Appellee is the city, not state, "
              "and Prosecutor is a city prosecutor, not county prosecutor. In your response, include whether the case "
              "meets standards or does not meet standards for future evaluation. If it does not meet standards for "
              "further evaluation, provide the reason why (from the above reasons listed). Please think through this "
              "task step-by-step. Use the following text from the case to make your decision:")
    # For every PDF in the directory of case PDFs, make a json object
    for i, pdf in enumerate(pdf_list, start=1):
        print(f"Processing {i}, {pdf}")
        data.append(to_jsonl(prompt, pdf_dir, pdf))
    return data

def to_json_ms(prompt, pdf_dir, pdf):
    """
    Creates json object for pdfs that meet standards
    ---json object format---
    {"Prompt": prompt, "Context": context, "Response": 'Meets Standards'}
    """
    context = read_pdf.extract_text(pdf_dir, pdf)[0]
    response = "Meets standards"
    return {"Prompt": prompt, "Context": context, "Response": response}

def to_json_dnms(prompt, pdf_dir, pdf):
    """
    Creates json object for pdfs that meet standards
    ---json object format---
    {"Prompt": prompt, "Context": context, "Response": 'Does not meet standards because [reasons]'}
    """
    df = pd.read_csv('Cases-With-Metadata.csv')
    metadata = read_pdf.extract_text(pdf_dir, pdf)[1]
    matching = df[df['Filename'] == pdf]
    reasons = ""
    if not matching.empty:
        row = df.loc[df['Filename'] == pdf]
        reasons = str(row['Standards Not Met'].values[0])
        reasons = reasons.replace("\"", " ")
    if matching.empty:
        print("No matches found")
    context = read_pdf.extract_text(pdf_dir, pdf)[0]
    response = "Does not meet standards because " + reasons
    return {"Prompt": prompt, "Context": context, "Response": response}

def to_json_dnms_questions(pdf_dir, pdf):
    """
    Creates json object for pdfs that do not meet standards
    ---json object format---
    {"Case": case index, "Context": context, "Standard 1": ... }
    """
    context = read_pdf.extract_text(pdf_dir, pdf)[0]
    index = pdf.replace(".pdf", "")
    print(index)
    df = pd.read_csv('cases_coded.csv')
    matching = df[df['Index'] == index]
    if not matching.empty:
        row = df.loc[df['Index'] == index]
        case_juv = str(row['case_juv'].values[0])
        case_crim = str(row['case_crim'].values[0])
        case_2001 = str(row['case_2001'].values[0])
        case_app = str(row['case_app'].values[0])
        case_pros = str(row['case_pros'].values[0])
        aoe_none = str(row['aoe_none'].values[0])
        aoe_grandjury = str(row['aoe_grandjury'].values[0])
        aoe_court = str(row['aoe_court'].values[0])
        aoe_defense = str(row['aoe_defense'].values[0])
        aoe_procbar = str(row['aoe_procbar'].values[0])
        aoe_prochist = str(row['aoe_prochist'].values[0])
        case_timeframe = str(row['case_timeframe'].values[0])
        other = str(row['other'])
    if matching.empty:
        print("No matches found")
    return {"Index": index, "Context": context, "case_juv": case_juv, "case_crim": case_crim, "case_2001": case_2001,
            "case_app": case_app, "case_pros": case_pros, "aoe_none": aoe_none, "aoe_grandjury": aoe_grandjury,
            "aoe_court": aoe_court, "aoe_defense": aoe_defense, "aoe_procbar": aoe_procbar, "aoe_prochist": aoe_prochist,
            "case_timeframe": case_timeframe, "other": other}

def to_json_ms_questions(pdf_dir, pdf):
    """
    Creates json object for pdfs that do meet standards
    ---json object format---
    {"Case": case index, "Context": context, "Standard 1": ... }
    """
    context = read_pdf.extract_text(pdf_dir, pdf)[0]
    index = pdf.replace(".pdf", "")
    print(index)
    return {"Index": index, "Context": context, "case_juv": 0, "case_crim": 0, "case_2001": 0,
            "case_app": 0, "case_pros": 0, "aoe_none": 0, "aoe_grandjury": 0,
            "aoe_court": 0, "aoe_defense": 0, "aoe_procbar": 0, "aoe_prochist": 0,
            "case_timeframe": 0, "other": 0}

def create_jsonl_train_test():
    """
    Create jsonl file for train and test sets
    """
    dnms_dir = "cases_pdf/DNMS_2018to2020"
    ms_dir = "cases_pdf/MS_2018to2020"
    data_dnms = process_to_jsonl(dnms_dir, to_json_dnms)
    data_ms = process_to_jsonl(ms_dir, to_json_ms)
    make_jsonl(data_ms, data_dnms)

def create_jsonl_inference():
    """
    Create jsonl file for inference
    """
    dnms_dir = "cases_pdf/DNMS_2021"
    ms_dir = "cases_pdf/MS_2021"
    data_dnms = process_to_jsonl(dnms_dir, to_json_dnms)
    data_ms = process_to_jsonl(ms_dir, to_json_ms)
    make_jsonl_inference(data_ms, data_dnms)




if __name__ == "__main__":
    dnms_dir = "../cases_pdf/DNMS"
    ms_dir = "../cases_pdf/MS"
    make_jsonl_questions(ms_dir, "../classify/jsonl/ms.jsonl", to_json_ms_questions)
