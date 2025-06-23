import pandas as pd



def eval_dnms(label, coded_df, results_df):
    filtered_df = coded_df[coded_df[label] == 1]
    idx_list = filtered_df["Index"].astype(str).tolist()

    total = 0
    correct = 0
    for index, row in results_df.iterrows():
        case = row["Index"]
        predicted = row["Predicted Label"]
        if case in idx_list:
            total += 1
            if predicted==1:
                correct += 1

    return total, correct, correct/total


def eval_explanation(label, label_explanation, coded_df, results_df):
    filtered_df = coded_df[coded_df[label] == 1]
    idx_list = filtered_df["Index"].astype(str).tolist()

    total = 0
    correct = 0
    for index, row in results_df.iterrows():
        case = row["Index"]
        explanation = row["Explanation"]
        if case in idx_list:
            total += 1
            if explanation==label_explanation:
                correct += 1

    return total, correct, correct/total


if __name__ == "__main__":
    coded_df = pd.read_csv("cases_coded.csv")
    results_df = pd.read_csv("aoe_questions_results_full_20250604_collapsed.csv")

    coded_df['aoe_none'] = coded_df[['aoe_none', 'aoe_court', 'aoe_defense']].any(axis=1).astype(int)
    print(coded_df['aoe_none'].head())

    print("AOE_PROCHIST\ndnms:")
    aoe_prochist_eval_dnms = eval_dnms("aoe_prochist", coded_df, results_df)
    print(aoe_prochist_eval_dnms)
    print("explanation:")
    aoe_prochist_eval_explanation = eval_explanation("aoe_prochist", "procedural history", coded_df, results_df)
    print(aoe_prochist_eval_explanation)

    print("AOE_PROCBAR\ndnms:")
    aoe_procbar_eval = eval_dnms("aoe_procbar", coded_df, results_df)
    print(aoe_procbar_eval)
    print("explanation:")
    aoe_procbar_eval_explanation = eval_explanation("aoe_procbar", "procedurally barred", coded_df, results_df)
    print(aoe_procbar_eval_explanation)

    print("AOE_NONE\ndnms:")
    aoe_none_eval = eval_dnms("aoe_none", coded_df, results_df)
    print(aoe_none_eval)
    print("explanation:")
    aoe_none_eval_explanation = eval_explanation("aoe_none", "not prosecutor", coded_df, results_df)
    print(aoe_none_eval_explanation)
