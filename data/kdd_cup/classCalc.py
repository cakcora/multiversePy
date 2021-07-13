f = open("kddcup_corrected.data", "r")
lines =  []
labelMap = {}
for x in f:
    line = x.split(",")
    try:
        labelMap[line[len(line)-1]] = labelMap[line[len(line)-1]] + 1
    except:
        labelMap[line[len(line)-1]] = 1
print(labelMap)
