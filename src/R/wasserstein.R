require(transport)
library(reshape)
results <- read.csv("https://raw.githubusercontent.com/cakcora/multiversePy/master/results/results_adult.csv")
data<-results[,c("depth","num_of_branches","dataset_name","feature_name")]
fdata<-data.frame()
# needs fixing - result file should contain one run per perturbed feature only.
for( dataset in unique(data$dataset_name)){
  for( feature in unique(data[data$dataset_name==dataset,]$feature_name)){
    data_f1 = data[data$dataset_name==dataset&data$feature_name==feature,]

    data_f1s = pp(as.matrix(data_f1[,c("depth","num_of_branches")]))
    for( feature2 in unique(data[data$dataset_name==dataset,]$feature_name)){

      data_f2 = data[data$dataset_name==dataset&data$feature_name==feature2,]

      data_f2s = pp(as.matrix(data_f2[,c("depth","num_of_branches")]))
      distance1d = wasserstein(data_f1s,data_f2s,p=1)
      distance2d = wasserstein(data_f1s,data_f2s,p=2)
      message(feature,"\t",feature2, "\t",distance1d,"\t",distance2d)
      fdata<-rbind(fdata,c(feature, feature2,  distance2d))

    }
  }
}


colnames(fdata)<-c("feature","feature2","distance")
fmatrix<-(cast(fdata, feature ~ feature2))
maxf = max(as.numeric(fdata$distance))
for( i in seq(2,length(fmatrix))){
  x=(fmatrix[,i])
  x=x[!is.na(x)]
  fv = as.numeric(x)/maxf
  m = median(fv,na.rm=TRUE)
  v = sqrt(var(fv,na.rm=TRUE))
  message(colnames(fmatrix)[[i]], " ",m," ",v)
}


require(ggplot2)
for( dataset in unique(data$dataset_name)){
for( feature in unique(data[data$dataset_name==dataset,]$feature_name)){
  if(feature=="Vanilla")next
  data3=data[data$feature_name%in%c("Vanilla",feature),]
  print(ggplot(data3,aes(x=depth,y=num_of_branches,color=feature_name))+geom_point()+ ggtitle(dataset))
}
}

