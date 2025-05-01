import os

dir = "../cases_olmocr/MS/results/"
with open("../cases_olmocr/MS/ms_olmocr_2.jsonl", "w") as out_f:
    for file in os.listdir(dir):
        with open(dir+file, "r") as in_f:
            out_f.write(in_f.read())
            out_f.write("\n")
            
