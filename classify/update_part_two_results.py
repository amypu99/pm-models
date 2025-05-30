import pandas as pd

# Load the CSV file
# df = pd.read_csv("./results/aoe_questions_results_sample_20250529.csv")

# # Convert 'result' to numeric in case it's read as string
# df['result'] = pd.to_numeric(df['result'], errors='coerce')

# # Group by 'index' and apply the aggregation rule
# aggregated = df.groupby('index')['result'].apply(lambda x: 1 if (x == 1).any() else 0).reset_index()

# # Save or display result
# aggregated.to_csv("./results/aggregated_results_20250529.csv", index=False)
# print(aggregated)



# # Load CSVs
grouped_df = pd.read_csv("./results/aggregated_results_20250529.csv")        # contains: index, result
pred_df = pd.read_csv("./results/pipeline_test_2025-05-26/aoe_procbar2.csv")               # contains: Index, Gold Label, Predicted Label, etc.

# Rename for consistency and clarity
grouped_df.rename(columns={'index': 'Index', 'result': 'Previous Result'}, inplace=True)

# Ensure numeric types
pred_df['Predicted Label'] = pd.to_numeric(pred_df['Predicted Label'], errors='coerce')
grouped_df['Previous Result'] = pd.to_numeric(grouped_df['Previous Result'], errors='coerce')
pred_df['Gold Label'] = pd.to_numeric(pred_df['Gold Label'], errors='coerce')
pred_df['Gold Label'] = pred_df['Gold Label'].map({0: 1, 1: 0})


# Merge on 'Index'
merged = pd.merge(pred_df, grouped_df, on='Index', how='inner')

# Compute final Result
merged['Result'] = merged.apply(
    lambda row: 1 if row['Predicted Label'] == 0 and row['Previous Result'] == 1 else 0,
    axis=1
)

# Extract desired columns
final_df = merged[['Index', 'Gold Label', 'Result']]

# Save and display
final_df.to_csv("./results/final_result.csv", index=False)
print(final_df)


# # df = pd.read_csv("./results/aoe_questions_results_sample_20250529.csv").groupby("index")
# # print(len(df))

# # df = pd.read_json("./results/extracted_evidence_sample.jsonl", lines=True).groupby("index")
# # print(len(df))