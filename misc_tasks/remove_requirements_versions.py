f = open("../requirements.txt", "r")

with open("../updated_requirements.txt", "w") as updated_file:
    for x in f:
        splitted = x.split("==")
        updated_file.write(splitted[0]+"\n")