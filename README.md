# PM with CJI

Code for the tasks associated with automating the extraction of information from prosecutorial misconduct cases in Ohio


## Directory Setup 
* ```cases_olmocr``` DNMS and MS cases from 2018-2021 converted to plaintext with AllenAI's Olmocr
* ```cases_pdf``` DNMS and MS case PDFs from 2018-2021 for classify/regex.py
* ```classify``` code for the full pipeline for classifying cases as MS and DNMS (see attached diagrams for pipeline logic)
    * ```classify/old``` code, results, and inputs from past classification attempts
    * ```classify/run_pipeline.py``` code to run whole pipeline (eventually)---currently only implements case-specific logic
    * ```classify/results``` results from full and partial pipeline runs
* ```read``` code for converting cases from PDFs to readable formats
* ```misc_tasks``` intermediary tasks not crucial to the full pipeline, i.e. updating requirements.txt for other machine


## Directory Structure
```
├── cases_olmocr
│   ├── DNMS
│   │   ├── dnms_olmocr_converted.jsonl
│   │   ├── dnms_olmocr.jsonl
│   │   └── test_dnms.jsonl
│   ├── MS
│       ├── ms_olmocr_converted.jsonl
│       └── ms_olmocr.jsonl
├── cases_pdf
│   ├── DNMS
│   ├── MS
├── classify
│   ├── aoe_logic.py
│   ├── filtered_jsonl/
│   ├── get_all_aoe.py
│   ├── old/
│   ├── regex.py
│   ├── results/
│   ├── run_baseline.py
│   ├── run_pipeline.py
│   └── run_questions.py 
├── misc_tasks/
├── read
│   ├── analyze_text.py
│   ├── cases_coded.csv
│   ├── gemini/
│   └── read_pdf.py
```
