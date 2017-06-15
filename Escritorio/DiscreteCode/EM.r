args<-commandArgs(TRUE)
#install.packages("mice", repos="http://cran.rstudio.com/")
read1<-function(dir, path){
  data<- read.csv(paste(dir, path, sep=""), header = FALSE, stringsAsFactors=FALSE, na.strings = "NaN", sep = ",")
  data<-matrix(unlist(data), ncol = ncol(data))
  data[is.nan(data)]<-NA

  return(data)
}

write1<-function(path, method, data){
  write(t(data), path, ncolumns = ncol(data), sep = ",")
}


EMBoot<-function(dir, path, m, part, data, k){
  
  suppressPackageStartupMessages(library(Amelia))
  
  set.seed(k)
  
  class<-data[,length(data[1,])]
  
  data<-data[,-length(data[1,])]
  
  malas<-c()
  values<-c()
  for(i in 1:length(data[1,])){
    print(values)
    print(i)
    if(2 > length(levels(factor(data[,i]))))
    {
      malas[length(malas)+1]<-i
      if(is.nan(levels(factor(data[,i]))[1]))
        values[length(values)+1]<-0
      else
        values[length(values)+1]<-as.integer(levels(factor(data[,i]))[1])
    }
  }
  
  print(malas)
  if(length(malas)>0)
    data<-data[,-malas]
  
  x<-length(data[,1])
  
  data<-matrix(as.numeric(data), nrow = x)
  
  a.out<-amelia(x = data, p2s = 0, m = 3, empri = 34)
  
  a<-a.out$imputations$imp1
  
  for (i in 2:m)
  {
    a<-a+a.out$imputations[[i]]
  }
  
  data<-a/m
  
  for(i in malas){
    if(i<2){
      data <- cbind(rep(0, length(data[1,])), data)
    }
    else{
      if(i>=length(data[1,])+length(malas)){
        data <- cbind(data[,1:length(data[1,])], rep(0, length(data[1,])))
      }
      else{
        data <- cbind(data[,1:i-1], rep(values[1], length(data[,1])), data[,(i):length(data[1,])])
        values<-values[-1]
      }
    }
  }
  
  return(cbind(data, class))
}
#DB, MD, Inst, Fold
i<-8;j<-3;k<-6;p<-1;lim<-376;dir<-"/home/unai/Escritorio/DiscreteCode/Data/"


i <- strtoi(args[1])
j <- strtoi(args[2])
k <- strtoi(args[3])
p <- strtoi(args[4])
lim <- strtoi(args[5])
dir <- args[6]

options(warn=-1)

path<-paste(i,"-",j, "-", k, "-", p, ".data", sep = "")

data<-read1(dir, path)
data1<-data[1:lim,]
data1<-EMBoot(dir, path, 3, p, data[1:lim,], k)
data2<-data[(lim+1):length(data[,1]),]
data2<-EMBoot(dir, path, 3, p, data[(lim+1):length(data[,1]),], k)

write1(paste(dir, i,"-",j, "-", k, "-", p, "-7", ".data", sep = ""), p, rbind(data1, data2))
