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


mice1<-function(dir, path, p, data){
  
  suppressPackageStartupMessages(library(mice))
  
  x<-length(data[,1])
  
  class<-data[,length(data[1,])]
  
  data<-data[,-length(data[1,])]
  
  data<-matrix(as.numeric(data), nrow = x)
  
  imp <- mice(data, print = FALSE)
  imp <- complete(imp)
  data<-matrix(unlist(imp), ncol = ncol(imp))
  
  return(cbind(data, class))
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

data1<-mice1(dir, path, p, data[1:lim,])
data2<-mice1(dir, path, p, data[(lim+1):length(data[,1]),])

write1(paste(dir, path), "6", p, rbind(data1, data2))

