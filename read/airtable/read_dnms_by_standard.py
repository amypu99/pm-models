import csv

def create_binary_reasons_csv(input_file, output_file):
    reasons = {
        "Defendant is a juvenile":"case_juv",
        "Case is not criminal": "case_crim",
        "Trial is before 2001": "case_2001",
        "Appellee is the city, not state": "case_app",
        "Prosecutor is a city pros., not county pros.": "case_pros",
        "No allegation of prosecutor misconduct": "aoe_none",
        "Misconduct is from a Grand Jury proceeding": "aoe_grandjury",
        "Allegation of error is against the court, not the prosecutor/state": "aoe_court",
        "Allegation is against defense attorney, not state": "aoe_defense",
        "Allegation is procedurally barred (e.g. claim is barred by res judicata)": "aoe_procbar",
        "Allegation is in procedural history": "aoe_prochist",
        "Appellate decision is outside of time frame (2017-2021)": "case_timeframe",
        "Other": "other"
    }

    cases_by_stnd= []
    with open(input_file, mode='r', encoding='utf-8-sig') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            cases_by_stnd.append(row)

    # Create output CSV
    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        # Header
        fieldnames = ["Index"] + list(reasons.values())
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for case in cases_by_stnd:
            row_dict = {"Index": case["Index"].strip()}
            standards_not_met = case.get("Which standard does it not meet?", "")

            for r in reasons.keys():
                row_dict[reasons[r]] = 1 if r in standards_not_met else 0
                if r=="Allegation of error is against the court, not the prosecutor/state" or r=="Allegation is against defense attorney, not state":
                   if row_dict[reasons[r]] == 1:
                       row_dict["aoe_none"] = 1

            writer.writerow(row_dict)

def main():
    input_file = "Cases-Does Not Meet Standards.csv"
    output_file = "../cases_coded.csv"
    create_binary_reasons_csv(input_file, output_file)

if __name__ == "__main__":
    main()