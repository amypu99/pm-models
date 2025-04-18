from run_baseline import *
from run_questions import *
from identify_allegations import *


def run_ordered():
    questions_dict = questions_setup()
    results_df = pd.DataFrame()

    filtered_jsonl = "jsonl/procbar_prochist_olmocr.jsonl"
    for q in questions:
        print(f"\nRunning question: {questions_dict[q]}")


        results_df = identify_allegations(jsonl_file=filtered_jsonl, question=aoe_procbar1_question,
                             label="aoe_procbar", label_func=label_answers)


        # Filter rows where Response Label == 0 for the next iteration
        filtered_jsonl = filter_jsonl(results_df)
        filtered_df_size = len(results_df[results_df['Response Label'] == 0])
        print(f"New q_df size after filtering: {filtered_df_size}")

        gc.collect()
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()