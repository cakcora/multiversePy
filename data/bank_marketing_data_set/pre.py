# ; -> ,
f = open("bank-full.csv", "r")
lines =  []
for x in f:
    corrected_line = ""
    corrected_line = (x.replace(";",",")).replace("\"","")
    lines.append(corrected_line)

f.close

f_new = open("bank-full-preprocessed.csv", "a")

for line in lines:
    f_new.write(line)

f_new.close()