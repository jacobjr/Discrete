args<-commandArgs(TRUE)

#install.packages("dunn.test")

options(warn=-1)


suppressPackageStartupMessages(library(dunn.test))
i<-1
m1<-c()
m2<-c()
while(args[i]!=0){
  m1[i]<-args[i]
  i<-i+1
}

i<-i+1
while(i<=length(args)){
  m2[i-length(m1)-1]<-args[i]
  i<-i+1
}

dunn.test(x = list(m1,m2), alpha = 0.05, kw = F, label = F, table = F)$P