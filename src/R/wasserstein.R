require(transport)
library(reshape)
require(ggplot2)
rm(list = ls())
num_trees=300
#results <- read.csv("https://raw.githubusercontent.com/cakcora/multiversePy/master/results/results_adult.csv")
datasets<-c("adult","chesskingrookvsking",
            "mushroom","nursery")
dir ="C:/Users/etr/PycharmProjects/multiversepy/results/"
for( dataset in datasets){
  results <- read.csv(paste0(dir,"results_",dataset,".csv"))
  data<-results[,c("depth","total_length_of_branches","dataset_name","feature_name")]
  fdata<-data.frame()
  for( feature in unique(data[data$dataset_name==dataset,]$feature_name)){
    data_f1 = data[data$dataset_name==dataset&data$feature_name==feature,]
    if(nrow(data_f1)!=300)
      quit(paste0("You do not have ",num_trees," trees"))
    data_f1s = pp(as.matrix(data_f1[,c("depth","total_length_of_branches")]))
    for( feature2 in unique(data[data$dataset_name==dataset,]$feature_name)){

      data_f2 = data[data$dataset_name==dataset&data$feature_name==feature2,]
      if(nrow(data_f2)!=300)
        quit(paste0("You do not have ",num_trees," trees"))
      data_f2s = pp(as.matrix(data_f2[,c("depth","total_length_of_branches")]))
      distance1d = wasserstein(data_f1s,data_f2s,p=1)
      distance2d = wasserstein(data_f1s,data_f2s,p=2)
      #message(feature,"\t",feature2, "\t",distance1d,"\t",distance2d)
      fdata<-rbind(fdata,c(feature, feature2,  distance2d))

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




  for( feature in unique(data$feature_name)){
    if(feature=="Vanilla")next

    data3=data[data$feature_name%in%c("Vanilla",feature),]
    f=paste0(dir,"figures/",dataset,feature,".jpg")
    p<-ggplot(data3,aes(x=total_length_of_branches,y=depth,color=feature_name))+geom_point()+ ggtitle(dataset)
    ggsave(filename=f,plot=p)

  }
}




