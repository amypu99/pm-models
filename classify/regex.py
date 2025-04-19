import os
import re
import json
from pypdf import PdfReader

directory = "../../cases_pdf_2"
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

                first_page_text = ""
                if len(reader.pages) > 0:
                    first_page_text = reader.pages[0].extract_text() or ""

                second_page_text = ""
                if len(reader.pages) > 1:
                    second_page_text = reader.pages[1].extract_text() or ""
                    
                all_text = first_page_text + "\n" + second_page_text

                juvenile_mentioned = bool(re.search(r"\bjuvenile\b", all_text, re.IGNORECASE))
                city_prosecutor = bool(re.search(r"\bCity\s+Prosecutor\b", all_text, re.IGNORECASE))

                match = standard_pattern.match(title)
                if not match:
                    # not even State v. … so DNMS and skip
                    predicted_label = "DNMS"
                    comment = "Does not match State v."
                else:
                    comment = "Title meets form. "

                    # check city prosecutor
                    if city_prosecutor:
                        predicted_label = "DNMS"
                        comment += "Prosecutor is a city prosecutor."
                    # check juvenile+initials
                    elif juvenile_mentioned and match.group(1).count('.') > 1:
                        predicted_label = "DNMS"
                        comment += "Juvenile mention and name in initials."
                    else:
                        predicted_label = "MS"
                        comment += "Neither city prosecutor nor juvenile."

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