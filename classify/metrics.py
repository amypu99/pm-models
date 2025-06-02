import sys
import pandas as pd

def compute_metrics(df):
    label_map = {0: "MS", 1: "DNMS"}
    df["Gold"]      = df["Gold Label"].map(label_map)
    df["Predicted"] = df["Predicted Label"].map(label_map)

    tp = ((df["Gold"] == "MS")   & (df["Predicted"] == "MS")).sum()
    tn = ((df["Gold"] == "DNMS") & (df["Predicted"] == "DNMS")).sum()
    fp = ((df["Gold"] == "DNMS") & (df["Predicted"] == "MS")).sum()
    fn = ((df["Gold"] == "MS")   & (df["Predicted"] == "DNMS")).sum()

    total = tp + tn + fp + fn
    accuracy  = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score  = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0)

    return {
        "true_positive":  tp,
        "true_negative":  tn,
        "false_positive": fp,
        "false_negative": fn,
        "accuracy":       accuracy,
        "precision":      precision,
        "recall":         recall,
        "f1_score":       f1_score
    }

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} path/to/your.csv")
        sys.exit(1)

    csv_path = sys.argv[1]
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)

    required_cols = {"Gold Label", "Predicted Label"}
    if not required_cols.issubset(df.columns):
        print(f"CSV must contain columns: {', '.join(required_cols)}")
        sys.exit(1)

    metrics = compute_metrics(df)
    for k, v in metrics.items():
        print(f"{k.replace('_', ' ').title():>15}: {v}")
