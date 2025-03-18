import json

def modify_aoe_none(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue  # skip empty lines

            record = json.loads(line)

            # Check the conditions
            if (record.get("aoe_defense") == "1" or
                record.get("aoe_none") == "1"    or
                record.get("aoe_court") == "1"):
                record["aoe_none"] = "1"

            # Skip the ones that we'd never get to through regex
            if (record.get("case_juv") == "1" or record.get("case_crim") == "1" or record.get("case_2001") == "1" or record.get("case_app") == "1" or record.get("case_pros") == "1"):
                continue
            # Write updated record
            fout.write(json.dumps(record) + "\n")

import matplotlib.pyplot as plt
import numpy as np

def split_and_measure(input_file, output_plot="boxplot.png"):
    """
    1. Reads the JSONL file, parsing one record per line.
    2. Splits the 'Context' field on the unicode paragraph separator '\\u00b6'.
    3. For each resulting segment, computes its character length.
    4. Collects all lengths and plots a box-and-whisker plot.
    5. Prints summary statistics including mean, median, std, percentiles, etc.
    """
    segment_lengths = []  # Will hold lengths of each split segment from all lines

    with open(input_file, 'r', encoding='utf-8') as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue  # Skip empty lines

            # Convert the JSON string to a Python dict
            record = json.loads(line)
            
            # Get the 'Context' field; default to an empty string if missing
            context_text = record.get("Context", "")
            
            # Split the context on the paragraph separator
            splits = context_text.split("\u00b6")
            
            # Measure the length (in characters) of each split
            for segment in splits:
                segment_lengths.append(len(segment))

    # Convert to NumPy array for easier numeric operations
    lengths_arr = np.array(segment_lengths, dtype=float)

    # -- Print summary statistics --
    if len(lengths_arr) == 0:
        print("No segments found. Exiting.")
        return

    count = len(lengths_arr)
    mean_val = np.mean(lengths_arr)
    median_val = np.median(lengths_arr)
    min_val = np.min(lengths_arr)
    max_val = np.max(lengths_arr)
    std_val = np.std(lengths_arr)
    pct1 = np.percentile(lengths_arr, 1)
    pct5 = np.percentile(lengths_arr, 5)
    pct10 = np.percentile(lengths_arr, 10)
    pct75 = np.percentile(lengths_arr, 75)
    pct90 = np.percentile(lengths_arr, 90)
    pct95 = np.percentile(lengths_arr, 95)
    pct99 = np.percentile(lengths_arr, 99)

    print("\nSummary Statistics for Segment Lengths:")
    print("---------------------------------------")
    print(f"Count:            {count}")
    print(f"Mean:             {mean_val:.2f}")
    print(f"Median:           {median_val:.2f}")
    print(f"Min:              {min_val}")
    print(f"Max:              {max_val}")
    print(f"Std Dev:          {std_val:.2f}")
    print(f"1th percentile:  {pct1}")
    print(f"5th percentile:  {pct5}")
    print(f"10th percentile:  {pct10}")
    print(f"75th percentile:  {pct75}")
    print(f"90th percentile:  {pct90}")
    print(f"95th percentile:  {pct95}")
    print(f"99th percentile:  {pct99}\n")

    # -- Plot a box-and-whisker (boxplot) of these lengths --
    # plt.figure(figsize=(8, 6))
    # plt.boxplot(lengths_arr, vert=True, patch_artist=True)
    # plt.title("Distribution of Segment Lengths (Split by \\u00b6)")
    # plt.ylabel("Segment length (characters)")
    
    # # Save the figure
    # plt.savefig(output_plot)
    # print(f"Boxplot saved as {output_plot}")
    # # If you want to display the plot interactively instead:
    # # plt.show()

if __name__ == "__main__":
    # modify_aoe_none("./dnms.jsonl", "./dnms_aoe_none.jsonl")
    split_and_measure("./dnms.jsonl", output_plot="boxplot.png")
