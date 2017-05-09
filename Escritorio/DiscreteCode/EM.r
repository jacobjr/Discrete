args<-commandArgs(TRUE)
#install.packages("mice", repos="http://cran.rstudio.com/")
read1<-function(dir, path){
  data<- read.csv(paste(dir, path, sep=""), header = FALSE, stringsAsFactors=FALSE, na.strings = "NaN", sep = " ")
  data<-matrix(unlist(data), ncol = ncol(data))
  data[is.nan(data)]<-NA
  
  malas<-c()
  for(i in 1:length(data[1,])){
    if(2 > length(levels(factor(data[,i]))))
      malas[length(malas)+1]<-i
  }
  print(malas)
  if(length(malas)>0)
    data<-data[,-malas]
  
  return(data)
}

write1<-function(path, method, part, data){
  
  dir<-strsplit(path, "[.]")
  dir = paste(unlist(dir)[1],"-", method, ".", unlist(dir)[2], sep = "")
  write(t(data), dir, ncolumns = ncol(data), sep = ",")
}


EMBoot<-function(dir, path, m, part, data){
  
  x<-length(data[,1])
  
  suppressPackageStartupMessages(library(Amelia))
  
  class<-data[,length(data[1,])]
  
  data<-data[,-length(data[1,])]
  
  data<-matrix(as.numeric(data), nrow = x)
  
  a.out<-amelia(x = data, p2s = 0, m = m, empri = 34)
  
  a<-a.out$imputations$imp1
  
  for (i in 2:m)
  {
    a<-a+a.out$imputations[[i]]
  }
  
  a<-a/m
  
  return(cbind(a, class))
}

i<-0;j<-0;k<-0;p<-0;lim<-276;dir<-"/home/unai/Escritorio/DiscreteCode/Data/"


i <- strtoi(args[1])
j <- strtoi(args[2])
k <- strtoi(args[3])
p <- strtoi(args[4])
lim <- strtoi(args[5])
dir <- args[6]

options(warn=-1)

path<-paste(i,"-",j, "-", k, "-", p, ".data", sep = "")

data<-read1(dir, path)

data1<-EMBoot(dir, path, 3, p, data[1:lim,])
data2<-EMBoot(dir, path, 3, p, data[(lim+1):length(data[,1]),])

write1(paste(dir, path), "7", p, rbind(data1, data2))
