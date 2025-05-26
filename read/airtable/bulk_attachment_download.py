import csv
import re
import requests
import os
from pypdf import PdfReader
from io import BytesIO

file_name = "Cases-All 2017 Cases.csv"
attachments_column_number = "Attachments"
index_column_name = "\ufeffIndex"
opinion_year_column_name = "Opinion year"
ms_column_name = "Does Not Meet Standards"
url_regex = r'(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?'

def parse_attachment(attachment_string):
    match = re.match(r'(.+?)\s*\((https?://[^)]+)\)', attachment_string)
    if match:
        return {
            'filename': match.group(1).strip(),
            'url': match.group(2).strip()
        }
    filename_only_match = re.match(r'(.+)', attachment_string)
    if filename_only_match:
        return {
            'filename': filename_only_match.group(1).strip(),
            'url': None
        }
    return None

def extract_metadata(url):
    if not url:
        return None
    try:
        response = requests.get(url)
        pdf_file = BytesIO(response.content)
        reader = PdfReader(pdf_file)
        return reader.metadata
    except Exception as e:
        print(f"Error reading metadata from {url}: {e}")
        return None

def download_file(url, filename, directory):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            file_path = os.path.join(directory, filename)
            with open(file_path, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded: {filename} to {directory}")
        else:
            print(f"Failed to download {filename}: Status code {response.status_code}")
    except Exception as e:
        print(f"Error downloading {filename} from {url}: {e}")

def run_extraction(results):
    for row_index, row in enumerate(results):
        # ms = row[ms_column_name]
        #
        # if ms is not None and ms != "":
        #     print(f"Skipping row {row_index + 1}/{len(results)}: MS column is not null")
        #     continue

        attachments = row[attachments_column_number].split(',')
        # opinion_year = int(row[opinion_year_column_name])
        citation_filename = row[index_column_name].strip()

        download_directory = "../../cases_pdf/MS_2017"
        # if opinion_year < 2021:
        #     download_directory = "../../cases_pdf/MS_2018to2020"
        # else:
        #     download_directory = "../../cases_pdf/MS_2021"

        os.makedirs(download_directory, exist_ok=True)

        for list_index, attachment in enumerate(attachments):
            attachment_info = parse_attachment(attachment)
            if not attachment_info:
                print(f"Could not parse attachment info from: {attachment}")
                continue

            url = attachment_info['url']
            if url:
                pdf_filename = f"{citation_filename}.pdf"
                download_file(url, pdf_filename, download_directory)
                print(f"✅ {list_index + 1}/{len(attachments)} from row {row_index + 1}/{len(results)}: {pdf_filename}")
                break
            else:
                print(f"No URL found for {attachment_info['filename']}")

def main():
    results = []
    with open(file_name, mode='r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            results.append(row)

    run_extraction(results)

if __name__ == "__main__":
    main()