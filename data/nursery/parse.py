f = open("nursery.data", "r")
lines =  []
broken_lines = []
labelMap = {}
for x in f:
    line = x.split(",")
    if (line[len(line) -1] == "not_recom\n" or line[len(line) -1] == "priority\n"):
        lines.append(x)
f.close


f_new = open("not_recom-priority-nursery.data", "a")

for line in lines:
    f_new.write(line)

f_new.close