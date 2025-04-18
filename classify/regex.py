import os
import re
import json
from pypdf import PdfReader

directory = "/Users/begumgokmen/Downloads/cases_pdf_2"
output_json = "labeled_results.json"

# Allows optional period after 'v' and handles leading whitespace
standard_pattern = re.compile(r"^\s*State v\.?\s*(.+)", re.IGNORECASE)

results = []

for gold_label in ["MS", "DNMS"]:
    subdirectory = os.path.join(directory, gold_label)
    if not os.path.isdir(subdirectory):
        print(f"Subdirectory '{subdirectory}' does not exist. Skipping.")
        continue

    for filename in os.listdir(subdirectory):
        file_path = os.path.join(subdirectory, filename)
        if os.path.isfile(file_path) and filename.lower().endswith('.pdf'):
            try:
                reader = PdfReader(file_path)
                meta = reader.metadata
                title = meta.title if meta.title else ""

                # If metadata title is empty, label as MS and skip processing.
                if title == "":
                    results.append({
                        "filename": filename,
                        "gold_label": gold_label,
                        "predicted_label": "MS",
                        "title": "",
                        "comment": "Empty metadata title; defaulted to MS."
                    })
                    continue

                # Check if title contains " v " or " v. " (case insensitive)
                if not (" v " in title.lower() or " v. " in title.lower()):
                    results.append({
                        "filename": filename,
                        "gold_label": gold_label,
                        "predicted_label": "MS",
                        "title": title,
                        "comment": "Title doesn't contain ' v ' or ' v.'; defaulted to MS."
                    })
                    continue

                # Extract text from first page
                first_page_text = ""
                if len(reader.pages) > 0:
                    first_page_text = reader.pages[0].extract_text() or ""

                # Juvenile check:
                # Match exactly the word "juvenile" (word boundaries), case-insensitive.
                juvenile_match = re.search(r"\bjuvenile\b", first_page_text, re.IGNORECASE)
                juvenile_mentioned = bool(juvenile_match)

                # Initialize predicted label and comment
                predicted_label = "MS"  # default label.
                comment = ""

                # Check if title meets the standard format "State v. Appellant"
                match = standard_pattern.match(title)
                if match:
                    comment += "Title meets form. "
                    appellant = match.group(1).strip()
                    # Check if the appellant name is in initials (assumed if contains more than one period).
                    appellant_is_initials = appellant.count('.') > 1

                    # Both conditions must be met: juvenile mentioned AND appellant name in initials.
                    if juvenile_mentioned and appellant_is_initials:
                        comment += "Juvenile mentioned and appellant name in initials."
                        predicted_label = "DNMS"
                    else:
                        comment += "Not juvenile or appellant name not in initials."
                        predicted_label = "MS"
                else:
                    comment += "Appellee not state; does not meet form."
                    predicted_label = "DNMS"

                results.append({
                    "filename": filename,
                    "gold_label": gold_label,
                    "predicted_label": predicted_label,
                    "title": title,
                    "comment": comment
                })

            except Exception as e:
                print(f"Error processing file {filename} in {gold_label}: {e}")

tp = sum(1 for r in results if r["gold_label"] == "MS" and r["predicted_label"] == "MS")
tn = sum(1 for r in results if r["gold_label"] == "DNMS" and r["predicted_label"] == "DNMS")
fp = sum(1 for r in results if r["gold_label"] == "DNMS" and r["predicted_label"] == "MS")
fn = sum(1 for r in results if r["gold_label"] == "MS" and r["predicted_label"] == "DNMS")

accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

metrics = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1_score,
    "true_positive": tp,
    "true_negative": tn,
    "false_positive": fp,
    "false_negative": fn
}

with open(output_json, mode='w', encoding='utf-8') as jsonfile:
    json.dump(results, jsonfile, indent=4)

print(metrics)