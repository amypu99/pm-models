import google.generativeai as genai
import os
import json
import time
import threading
from datetime import datetime
from collections import deque
import numpy as np


class RateLimiter:
    def __init__(self):
        self.rpm_queue = deque(maxlen=15)
        self.daily_requests = 0
        self.last_reset_date = datetime.now().date()
        self.lock = threading.Lock()

    def can_make_request(self):
        with self.lock:
            current_time = datetime.now()
            current_date = current_time.date()

            if current_date > self.last_reset_date:
                self.daily_requests = 0
                self.last_reset_date = current_date

            while self.rpm_queue and (current_time - self.rpm_queue[0]).total_seconds() > 60:
                self.rpm_queue.popleft()

            if (len(self.rpm_queue) < 15 and
                    self.daily_requests < 1500):
                return True
            return False

    def record_request(self):
        with self.lock:
            self.rpm_queue.append(datetime.now())
            self.daily_requests += 1

def process_all(process_func, pdf_directory, json_directory, prompt_path, error_log_path, rate_limiter, model):
    """Process one PDF file."""
    processed_files = set(f.replace('.json', '.pdf') for f in os.listdir(json_directory) if f.endswith('.json'))
    # processed_error_files = set(f.replace('.json', '.pdf') for f in os.listdir("../cases_json/json_MS_errors") if f.endswith('.json'))

    pdf_list = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf') and f not in processed_files]

    print(f"Found {len(pdf_list)} PDFs to process")

    for i, pdf in enumerate(pdf_list, start=1):
        print(i)
        process_func(pdf_directory, json_directory, pdf, prompt_path, error_log_path, rate_limiter, model)
        time.sleep(4)




def process_pdf(pdf_directory, json_directory, pdf, prompt_path, error_log_path, rate_limiter, model):
    """Process one PDF file in pdf_directory with given prompt"""
    process_successful = False
    response = None

    pdf_path = pdf_directory + '/' + pdf
    filename = pdf.strip(".pdf")

    while not rate_limiter.can_make_request():
        time.sleep(1)

    try:
        print(f"Processing {filename}")
        case_pdf = genai.upload_file(pdf_path)
        with open(prompt_path, "r") as file:
            prompt = file.read().strip()

        rate_limiter.record_request()
        response = model.generate_content([prompt, case_pdf])

        # Ensure output directory exists
        os.makedirs(json_directory, exist_ok=True)

        if response and response.text:
            os.makedirs(json_directory, exist_ok=True)
            print(f"Successfully processed {filename}")
            process_successful = True

    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")
        with open(error_log_path, "a") as error_log:
            error_log.write(f"{datetime.now()}: Error processing {filename}: {str(e)}\n")

    if process_successful and response and response.text:
        with open(f"{json_directory}/{filename}.json", "w") as out_file:
            json.dump(response.text, out_file, indent=6)



def randomize():
    processed_files = [f.replace('.json', '.pdf') for f in os.listdir("../../cases_json/json_DNMS") if f.endswith('.json')]
    random = np.random.choice(processed_files, 10)
    print(random)


if __name__ == "__main__":
    genai.configure(api_key='AIzaSyChB60ef0mrYKoY2UobtM-fWqS-gWsHoJY')
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")

    pdf_dir = "cases_pdf/MS"
    json_dir = "cases_json/json_MS_errors"
    prompt = "read/readpdf_prompt.txt"
    error = "read/error_logMS2.txt"
    rate_limiter = RateLimiter()


    process_all(process_func=process_pdf, pdf_directory=pdf_dir, json_directory=json_dir, prompt_path=prompt,
                error_log_path=error, rate_limiter=rate_limiter, model=model)
