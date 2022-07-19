setwd("C:\\work\\internship\\multiversePy\\data\\")

df = read.table("BankMarketing/bank-additional-full.txt", sep = ";", header = TRUE)
write.csv(df, "BankMarketing/bank-additional-full-process.csv", row.names = FALSE)


df = read.table("WineQuality/winequality-white.csv", sep = ";", header = TRUE)
write.csv(df, "WineQuality/winequality-white-process.csv", row.names = FALSE)

df = read.delim2("Yeast/yeast.data", header = FALSE, sep="", stringsAsFactors = FALSE)
names(df) <- c("sequenceName","mcq", "gvh", "alm", "mit", "erl", 
               "pox", "vac", "nuc", "label")
write.csv(df[,2:10], "Yeast/yeast.csv", row.names = FALSE)


df = read.delim2("StateLog/sat.trn", header = FALSE, sep="", stringsAsFactors = FALSE)
write.csv(df, "StateLog/sat.csv", row.names = FALSE)

df = read.delim2("PolishCompaniesBankruptcy/1year.txt", header = FALSE, sep=",", stringsAsFactors = FALSE)
#write.csv(df, "StateLog/sat.csv", row.names = FALSE)

df = read.delim2("SensorlessDriveDiagnosis/Sensorless_drive_diagnosis.txt", header = FALSE, sep="", stringsAsFactors = FALSE)
#names(df) <- c("sequenceName","mcq", "gvh", "alm", "mit", "erl", 
#               "pox", "vac", "nuc", "label")
write.csv(df, "SensorlessDriveDiagnosis/Sensorless_drive_diagnosis.csv", row.names = FALSE)

##SteelPlateFault dataset, add header and convert the label columns into one column
df = read.delim2("SteelPlateFault/Faults.NNA", header = FALSE, sep="", stringsAsFactors = FALSE)
header = read.csv("SteelPlateFault/Faults27x7_var")
df2 = df[,28:34]
class_vec = rep(0, nrow(df))
for(i in 1:nrow(df)){
  class_vec[i] = which(df2[i,]==1)
}
df$label <- class_vec
df <- df[,c(1:27,35)]
names(df)[1:27] <- t(header)[1:27]
write.csv(df, "SteelPlateFault/Faults.csv", row.names = FALSE)

#
df = read.delim2("Biodegradation/biodeg.csv", header = FALSE, sep=";", stringsAsFactors = FALSE)
write.csv(df, "Biodegradation/biodeg_process.csv", row.names = FALSE, col.names = FALSE)


df = read.delim2("FirmTeacherClaveDirection/ClaveVectors_Firm-Teacher_Model.txt", header = FALSE, sep="", stringsAsFactors = FALSE)
df2 = df[,17:20]
class_vec = rep(9, nrow(df))

for(i in 1:nrow(df)){
  xx = which(df2[i,]==1)
  if(length(xx)==1){
    class_vec[i] = xx
  }
}
df$Class <- class_vec
df <- df[,c(1:16,21)]
df <- df[df$Class != 9,]
write.csv(df, "FirmTeacherClaveDirection/ClaveVectors_Firm-Teacher_Model.csv", row.names = FALSE)



df = read.delim2("PolishCompaniesBankruptcy/1year.arff", header = FALSE, sep=",", stringsAsFactors = FALSE)
names(df)[ncol(df)] <- "Class"
write.csv(df, "PolishCompaniesBankruptcy/1year.csv", row.names = FALSE)



df = read.delim2("PhishingWebsites/PhishingWebsites.txt", header = FALSE, sep=",", stringsAsFactors = FALSE)
header = read.delim2("PhishingWebsites/Training Dataset.arff", sep="", header = FALSE)[,2]
names(df) <- header
names(df)[ncol(df)] <- "Class"
write.csv(df, "PhishingWebsites/PhishingWebsites.csv", row.names = FALSE)


df = read.delim2("RecognitionHandWrittenDigits/optdigits.tra", header = FALSE, sep=",", stringsAsFactors = FALSE)
names(df)[ncol(df)] <- "Class"
write.csv(df, "RecognitionHandWrittenDigits/optdigits.csv", row.names = FALSE)







