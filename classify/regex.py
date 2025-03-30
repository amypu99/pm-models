import os
import re
import json
from pypdf import PdfReader

directory = "/pm-models/classify/cases_pdf_2"

output_json = "labeled_results.json"

pattern = re.compile(r"^State v\.\s*(.+)", re.IGNORECASE)

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
                
                # initialize predicted label and comment
                predicted_label = "MS"  # default to MS (false positives prioritized)
                comment = ""
                
                # check if title meets the standard format "State v. Appellant"
                match = pattern.match(title)
                if match:
                    comment += "Title meets form."
                    # extract appellant name
                    appellant = match.group(1).strip()
                    # check juvenile: if more than one period in appellant, label as DNMS.
                    if appellant.count('.') > 1:
                        comment = "Juvenile (appellant name in initials)."
                        predicted_label = "DNMS"
                    else:
                        comment += " No error."
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

# store labeled results
with open(output_json, mode='w', encoding='utf-8') as jsonfile:
    json.dump(results, jsonfile, indent=4)

print(metrics)