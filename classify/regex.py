import os
import re
import json
from pypdf import PdfReader
import pandas as pd


def identify_regex_dnms(directory):
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

            index_val = filename.replace(".pdf", "")

            if os.path.isfile(file_path) and filename.lower().endswith('.pdf'):
                try:
                    reader = PdfReader(file_path)
                    meta = reader.metadata
                    title = meta.title if meta.title else ""

                    # If metadata title is empty, label as MS and skip processing.
                    if title == "":
                        results.append({
                            "Index": index_val,
                            "Gold Label": gold_label,
                            "Predicted Label": 0,
                            "Title": title,
                            "Comment":  "Empty metadata title; defaulted to MS."
                        })
                        continue

                    # Check if title contains " v " or " v. " (case insensitive)
                    if not (" v " in title.lower() or " v. " in title.lower()):
                        results.append({
                            "Index": index_val,
                            "Gold Label": gold_label,
                            "Predicted Label": 0,
                            "Title": title,
                            "Comment": "Title doesn't contain ' v ' or ' v.'; defaulted to MS."
                        })
                        continue

                    # Extract text from first and second page
                    first_page_text = ""
                    if len(reader.pages) > 0:
                        first_page_text = reader.pages[0].extract_text() or ""
                    second_page_text = ""
                    if len(reader.pages) > 0:
                        second_page_text = reader.pages[1].extract_text() or ""

                    all_text = first_page_text + '\n'+ second_page_text

                    text = meta.title or all_text
                    match = re.match(r"^\s*State v\.?\s*(.+)", text, re.IGNORECASE)
                    # Juvenile check:
                    juvenile_mentioned = bool(re.search(r"\bjuvenile\b", all_text, re.IGNORECASE))
                    # City prosecutor check:
                    city_prosecutor = bool(re.search(r"\bCity\s+Prosecutor\b", all_text, re.IGNORECASE))
                    municipal_court = bool(re.search(r'\bmunicipal court\b', all_text, re.IGNORECASE))

                    # Initialize predicted label and comment
                    predicted_label = 0  # default label.
                    comment = ""
                    if not match:
                        predicted_label = 1
                        comment = "Does not match State v."
                    elif city_prosecutor:
                        predicted_label = 1
                        comment = "Title meets form. Prosecutor is a city prosecutor."
                    elif municipal_court:
                        predicted_label = 1
                        comment = "Title meets form. Case originated in a Municipal Court (city prosecutor)."
                    elif juvenile_mentioned and match.group(1).count('.') > 1:
                        predicted_label = 1
                        comment = "Title meets form. Juvenile mention and name in initials."
                    else:
                        predicted_label = 0
                        comment = "Title meets form. Neither city prosecutor, municipal court, nor juvenile."

                    results.append({
                        "Index": index_val,
                        "Gold Label": gold_label,
                        "Predicted Label": predicted_label,
                        "Title": title,
                        "Comment": comment
                    })

                except Exception as e:
                    print(f"Error processing file {filename} in {gold_label}: {e}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"./pipeline_test/regex.csv", index=False)

    return results_df


def evaluate_regex(results):
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
    return metrics

if __name__ == "__main__":
    directory = "/Users/begumgokmen/Downloads/cases_pdf_2"
    results = identify_regex_dnms(directory)
    evaluate_regex(results)
