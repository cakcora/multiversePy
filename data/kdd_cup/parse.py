f = open("kddcup.data.corrected", "r")
lines =  []
broken_lines = []
labelMap = {}
for x in f:
    line = x.split(",")
    if (line[len(line) -1] == "normal.\n" or line[len(line) -1] == "smurf.\n"):
        lines.append(x)
f.close


f_new = open("kddcup_corrected.data", "a")

for line in lines:
    f_new.write(line)

f_new.close