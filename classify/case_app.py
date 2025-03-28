import pandas as pd
import os
import re
from run_questions import load_jsonl


def check_text_files_for_pattern(df, folder_path="../cases_txt/MS"):
    if "Response Label" not in df.columns:
        df["Response Label"] = None

    # Ensure folder path exists
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder path {folder_path} does not exist")

    # Get list of files in the folder
    files_in_folder = os.listdir(folder_path)

    # Process each row in the DataFrame
    for idx, row in df.iterrows():
        # Get the index string
        index_str = str(row["Index"])
        file_name = f"{index_str}.txt"

        # Check if the file exists in the folder
        if file_name in files_in_folder:
            file_path = os.path.join(folder_path, file_name)

            # Read the file content
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()

                content_lower = content.lower()

                # Check if "City of" comes before "Appellee" in the text
                city_matches = [m.start() for m in re.finditer(r'city of', content_lower)]
                appellee_matches = [m.start() for m in re.finditer(r'appellee', content_lower)]

                if city_matches and appellee_matches:
                    # If both phrases exist, check if any "city of" comes before any "appellee"
                    if min(city_matches) < min(appellee_matches):
                        df.at[idx, "Response Label"] = 1
                    else:
                        df.at[idx, "Response Label"] = 0
                else:
                    # If one or both phrases don't exist, mark as 0
                    df.at[idx, "Response Label"] = 0
            except Exception as e:
                print(f"Error processing file {file_name}: {str(e)}")
                df.at[idx, "Response Label"] = 0
        else:
            # If file doesn't exist, leave as None or set to a default value
            print(f"File {file_name} not found in {folder_path}")

    return df


# Example usage
if __name__ == "__main__":

    # df = pd.read_csv("standards_csv/case_app.csv")
    df = load_jsonl("jsonl/ms.jsonl")

    result_df = check_text_files_for_pattern(df)
    result_df.to_csv("standards_csv/ms_case_app_regex.csv", index=False)

    print("Processing complete!")