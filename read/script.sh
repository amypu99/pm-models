#!/usr/bin/env bash

# Loop over every .pdf file in cases_temp/DNMS
for file in cases_temp/MS/*.pdf
do
  # Run the Python pipeline command on the current file
  python3 -m olmocr.pipeline ./cases_olmocr_2/MS --pdfs "$file"
done