import os
import re
import json
from pypdf import PdfReader
import pandas as pd

def identify_regex_dnms(directory):
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
                    index_val = filename.replace(".pdf", "")

                    if title == "":
                        results.append({
                            "Index": index_val,
                            "Gold Label": gold_label,
                            "Predicted Label": 0,
                            "Title": title,
                            "Comment": "Empty metadata title; defaulted to MS."
                        })
                        continue

                    if not (" v " in title.lower() or " v. " in title.lower()):
                        results.append({
                            "Index": index_val,
                            "Gold Label": gold_label,
                            "Predicted Label": 0,
                            "Title": title,
                            "Comment": "Title doesn't contain ' v ' or ' v.'; defaulted to MS."
                        })
                        continue

                    first_page_text = reader.pages[0].extract_text() if reader.pages else ""
                    second_page_text = reader.pages[1].extract_text() if len(reader.pages) > 1 else ""
                    first_two = first_two = (first_page_text or "") + "\n" + (second_page_text or "")
                    text = "".join((page.extract_text() or "") + "\n" for page in reader.pages)

                    match = re.match(r"^\s*State v\.?\s*(.+)", title, re.IGNORECASE)
                    
                    juvenile_mentioned = bool(re.search(r"\bjuvenile\b", first_two, re.IGNORECASE))
                    juvenile_phrase = bool(re.search(r"\bjuvenile court\b.*\btransfer jurisdiction to\b", first_two, re.IGNORECASE))
                    
                    city_prosecutor = bool(re.search(r"\bcity\b.*\bprosecutor\b", text, re.IGNORECASE))
                
                    municipal = bool(re.search(r"\bmunicipal\s+court\b", text, re.IGNORECASE))
                    county_pros1 = bool(re.search(r"\bcounty\s+prosecutor(?:'s)?\b", first_two, re.IGNORECASE))
                    county_pros2 = bool(re.search(r"\bcounty\s+prosecuting\s+attorney(?:'s)?\b", first_two, re.IGNORECASE))
                    county_pros3 = bool(re.search(r"\bspecial\b.*\bprosecutors\b", first_two, re.IGNORECASE))
                    common_pleas = bool(re.search(r"\bcommon\s+pleas\b", first_two, re.IGNORECASE))
                    county_level = county_pros1 or county_pros2 or county_pros3 or common_pleas
                    municipal_court = municipal and not county_level

                    predicted_label = 0 
                    comment = ""

                    if not match:
                        predicted_label = 1
                        comment = "Does not match State v."
                    elif city_prosecutor and not common_pleas:        
                        predicted_label = 1
                        comment = "Prosecutor is a city prosecutor."
                    elif municipal_court:
                        predicted_label = 1
                        comment = "Case originated in a municipal court."
                    elif (juvenile_mentioned and match.group(1).count('.') > 1) and (not juvenile_phrase):
                        predicted_label = 1
                        comment = "Case is juvenile."
                    else:
                        predicted_label = 0
                        comment = "Case meets standards."
                        
                    results.append({
                        "Index": filename,
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