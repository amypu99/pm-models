import pandas as pd
import os
import glob
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def evaluate_csv_file(file_path):
    """
    Evaluate a single CSV file by comparing the filename-based column to the Response Label.
    """
    try:
        base_filename = os.path.basename(file_path).replace('.csv', '')
        df = pd.read_csv(file_path)

        if base_filename not in df.columns or 'Response Label' not in df.columns:
            print(f"Skipping {file_path} - Required columns not found")
            return None

        # Drop uncertain labels (99)
        df = df[(df[base_filename] != 99) & (df['Response Label'] != 99)]

        gold_labels = df[base_filename].dropna().astype(int).tolist()
        response_labels = df['Response Label'].dropna().astype(int).tolist()

        if len(gold_labels) != len(response_labels) or not gold_labels:
            print(f"Skipping {file_path} - Mismatched or empty labels")
            return None

        metrics = {
            'filename': base_filename,
            'accuracy': accuracy_score(gold_labels, response_labels),
            'precision': precision_score(gold_labels, response_labels, average='weighted', zero_division=0),
            'recall': recall_score(gold_labels, response_labels, average='weighted', zero_division=0),
            'f1_score': f1_score(gold_labels, response_labels, average='weighted', zero_division=0)
        }

        return metrics

    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None


def evaluate_all_csvs_in_folder(folder_path="standards_csv"):
    """
    Process all CSV files in the specified folder.
    """
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    if not csv_files:
        print(f"No CSV files found in {folder_path}")
        return

    results = []
    for file_path in csv_files:
        print(f"Processing {os.path.basename(file_path)}...")
        metrics = evaluate_csv_file(file_path)
        if metrics:
            results.append(metrics)

    if results:
        results_df = pd.DataFrame(results)
        output_file = "evaluation_results.csv"
        results_df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
    else:
        print("No files successfully evaluated.")


if __name__ == "__main__":
    evaluate_all_csvs_in_folder("./")
