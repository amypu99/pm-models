"""
**PDF to text**
This script extracts the text from case PDFs
Assumes downloaded: cases_pdf/MS_2018_2021, cases_pdf/DNMS_2018_2021,
Dependencies: os, pypdf
"""
import os
from pypdf import PdfReader

def process_to_txt(pdf_dir, output_dir, to_txt):
    """
    Creates txt files in output_dir for each PDF in pdf_dir
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
    Creates txt files in output_dir for 1 PDF
    """
    filename = pdf.strip("pdf")
    output_path = output_dir + '/' + filename + 'txt'
    text = extract_text(pdf_dir, pdf)[0]
    with open(output_path, "w") as out_file:
        out_file.write(text)

def extract_text(pdf_dir, pdf):
    """
    Extracts text from PDF
    """
    pdf_path = pdf_dir + '/' + pdf
    reader = PdfReader(pdf_path)
    case_text = ""
    for page in reader.pages:
        case_text += page.extract_text()
    return case_text, reader.metadata



if __name__ == "__main__":
    ms_pdf_dir = "../cases_pdf/MS_2017"
    ms_output_dir = "/Users/prishasamdarshi/Documents/School/Year 5/DL/cases_txt/MS_2017"
    process_to_txt(ms_pdf_dir, ms_output_dir, to_txt)
    # dnms_pdf_dir = "../cases_pdf/DNMS"
    # dnms_output_dir = "../cases_txt/DNMS"
    # process_to_txt(dnms_pdf_dir, dnms_output_dir, to_txt)



