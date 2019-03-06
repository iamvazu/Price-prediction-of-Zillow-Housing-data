# Price-prediction-of-Zillow-Housing-data
Analyze the dataset on Zillow housing obtained from Kaggle. Perform Predictive Analytics by predicting the price of house using Machine learning models such as linear models, tree models and non-linear models 
#Price prediction for Zillow housing data
library(ggplot2)
library(tidyverse)
library(corrplot)
library(AppliedPredictiveModeling)
library(caret)
library(e1071)
library(stringr)
library(imputeTS)
library(DMwR)
library(mice)
library(pls)
library(elasticnet)
library(nnet)
library(kernlab)
library(randomForest)
library(gbm)

#reading the data
data_2016<-read.csv("properties_2016.csv",quote="") 
data_2016<-data_2016[,-c(56:59)] # unwanted columns removed
data_2017<-read.csv("properties_2017.csv", quote = "") 
data_2017<-data_2017[,-c(56:59)] # unwanted columns removed

#combining data
data<-rbind(data_2016,data_2017)

#removing unnecessary columns
filtered_data<-data[,-c(9,10,11,12:17,20,25,26,29:32,35,38,41:44,46,47,49)]
filtered_data$fireplaceflag<-as.factor(filtered_data$fireplaceflag)

#imputations na to 0
filtered_data$architecturalstyletypeid <- na.replace(filtered_data$architecturalstyletypeid , 0)
filtered_data$basementsqft <- na.replace(filtered_data$basementsqft , 0)
filtered_data$buildingclasstypeid <- na.replace(filtered_data$buildingclasstypeid , 0)
filtered_data$garagecarcnt <- na.replace(filtered_data$garagecarcnt , 0)
filtered_data$poolcnt <- na.replace(filtered_data$poolcnt , 0)
filtered_data$taxamount <- na.replace(filtered_data$taxamount , 0)

#removing rows for variables where there are very few NAs
rows1<-which(filtered_data$propertycountylandusecode=="")
filtered_data<-filtered_data[-rows1,]
rows2<-which(is.na(filtered_data$regionidcity))
filtered_data<-filtered_data[-rows2,]
rows3<-which(is.na(filtered_data$regionidzip))
filtered_data<-filtered_data[-rows3,]
rows4<-which(is.na(filtered_data$yearbuilt))
filtered_data<-filtered_data[-rows4,]

#imputing tuborspa
rows12<-which(is.na(filtered_data$hashottuborspa))
#filtered_data$hashottuborspa[rows12]<-"FALSE"

#fireplace column test
fireplacecount<-which(filtered_data$fireplaceflag!="")
filtered_data$fireplacecnt[fireplacecount]<-"FALSE"
filtered_data<-filtered_data[,-25] #removing fireplaceflag column after imputing fireplacecount

#garagetotalsqft imputation
rows6<-which(is.na(filtered_data$garagetotalsqft))
filtered_data$garagetotalsqft[rows6]<-median(filtered_data$garagetotalsqft[-rows6])

#row count with nas
aircon_rows<-which(is.na(filtered_data$airconditioningtypeid))
arc_rows<-which(is.na(filtered_data$architecturalstyletypeid))
builqlty_rows<-which(is.na(filtered_data$buildingqualitytypeid))
heating_rows<-which(is.na(filtered_data$heatingorsystemtypeid))
hastub_rows<-which(filtered_data$hashottuborspa!="")
regneigbourhood_rows<-which(is.na(filtered_data$regionidneighborhood))
unicnt_rows<-which(is.na(filtered_data$unitcnt))
fire_rows<-which(is.na(filtered_data$fireplacecnt))

#imputing random values for airconditioningtypeid
filtered_data$airconditioningtypeid[is.na(filtered_data$airconditioningtypeid)]<-floor(runif(4179041, min=0, max=14))

#imputing building quality type id
filtered_data$buildingqualitytypeid[is.na(filtered_data$buildingqualitytypeid)]<-floor(rnorm(1951764, mean=5, sd=1.66))

#imputing heatingsystemtypeid
filtered_data$heatingorsystemtypeid[is.na(filtered_data$heatingorsystemtypeid)]<-floor(runif(2152669, min=0, max=26))

#imputing unitcnt
filtered_data$unitcnt[is.na(filtered_data$unitcnt)]<-floor(runif(1877778, min=0, max=6))

#imputing fireplacecnt
filtered_data$fireplacecnt<-as.numeric(filtered_data$fireplacecnt)
filtered_data$fireplacecnt[is.na(filtered_data$fireplacecnt)]<-floor(runif(5138983, min=0, max=4))

#removing hashottuborspa
filtered_data<-filtered_data[,-13]

# split data based on fips county code
LAdata<-filtered_data[which(filtered_data$fips==6037),]
orangedata<-filtered_data[which(filtered_data$fips==6059),]
venturadata<-filtered_data[which(filtered_data$fips==6111),]

#lotsqft imputation
rows5<-which(is.na(LAdata$lotsizesquarefeet))
LAdata$lotsizesquarefeet[rows5]<-median(LAdata$lotsizesquarefeet[-rows5])
rows7<-which(is.na(orangedata$lotsizesquarefeet))
orangedata$lotsizesquarefeet[rows7]<-median(orangedata$lotsizesquarefeet[-rows7])
rows8<-which(is.na(venturadata$lotsizesquarefeet))
venturadata$lotsizesquarefeet[rows8]<-median(venturadata$lotsizesquarefeet[-rows8])

#imputaion of structuredollarvaluecnt
rows9<-which(is.na(LAdata$structuretaxvaluedollarcnt))
LAdata$structuretaxvaluedollarcnt[rows9]<-median(LAdata$structuretaxvaluedollarcnt[-rows9])
rows10<-which(is.na(orangedata$structuretaxvaluedollarcnt))
orangedata$structuretaxvaluedollarcnt[rows10]<-median(orangedata$structuretaxvaluedollarcnt[-rows10])
rows11<-which(is.na(venturadata$structuretaxvaluedollarcnt))
venturadata$structuretaxvaluedollarcnt[rows11]<-median(venturadata$structuretaxvaluedollarcnt[-rows11])

#imputaion of landtaxvaluedollarvaluecnt
rows13<-which(is.na(LAdata$landtaxvaluedollarcnt))
LAdata$landtaxvaluedollarcnt[rows13]<-median(LAdata$landtaxvaluedollarcnt[-rows13])
rows14<-which(is.na(orangedata$landtaxvaluedollarcnt))
orangedata$landtaxvaluedollarcnt[rows14]<-median(orangedata$landtaxvaluedollarcnt[-rows14])
rows15<-which(is.na(venturadata$landtaxvaluedollarcnt))
venturadata$landtaxvaluedollarcnt[rows15]<-median(venturadata$landtaxvaluedollarcnt[-rows15])

#calculating taxvaluedollarcnt
LAdata$taxvaluedollarcnt<-LAdata$structuretaxvaluedollarcnt+LAdata$landtaxvaluedollarcnt
orangedata$taxvaluedollarcnt<-orangedata$structuretaxvaluedollarcnt+orangedata$landtaxvaluedollarcnt
venturadata$taxvaluedollarcnt<-venturadata$structuretaxvaluedollarcnt+venturadata$landtaxvaluedollarcnt

#code for scree plot
newdata<-rbind(LAdata,orangedata,venturadata)
newdata<-newdata[,-c(1,3,4,7,9,13,15:18,21,22,26)]
newdata<-newdata[,-10]
newdata<-newdata[,-c(13,14)]
str(newdata)
segPCA<-prcomp(newdata)
percentVariancePCA = segPCA$sd^2/sum(segPCA$sd^2)*100
plot(percentVariancePCA, xlab="Component", ylab="Percentage of Total Variance", type="l", main="PCA")

#VENTURA 

venturadata_predictor<-venturadata[,-25]
venturadata_response<-venturadata[,25]

#finding columns with zero variance
zerovar_ventura<-nearZeroVar(venturadata_predictor)

#removing columns with zero variance
venturadata_nozerovar<-venturadata[,-zerovar_ventura]

#finding categorical data in ventura
str(venturadata_nozerovar)

#removing factor data
venturadata_nofactor<-venturadata_nozerovar[,-c(12,14)]

#correlation of ventura_data
corr_ventura<-cor(venturadata_nofactor)
cutoff_ventura<-findCorrelation(corr_ventura,cutoff=0.8)
venturadata_final<-venturadata_nofactor[,-cutoff_ventura]
venturadata_final$taxamount<-venturadata_nofactor$taxamount #adding taxamount
venturadata_final<-venturadata_final[,-c(1,11)] #removing unwanted columns(parcel id, poolcnt(too many zeros))

#removing columns with high correlation
venturadata_final<-venturadata_final[,-13] 

#removing low coerrelation columns
venturadata_final<-venturadata_final[,-c(1,4,8,13)]
venturadata_final$structuredollarvaluecnt<-venturadata$structuretaxvaluedollarcnt
venturadata_final$landtaxvaluedollarcnt<-venturadata$landtaxvaluedollarcnt
venturadata_final$taxamount<-venturadata_nofactor$taxamount
venturadata_final$price<-venturadata_response
ventura_preprocess<-preProcess(venturadata_final,method=c("BoxCox","center","scale","pca"),
                               thresh=0.95,freqCut=20,uniqueCut=10,cutoff=0.8,fudge=0.2,numUnique=30)
ventura_preprocessresults<-predict(ventura_preprocess,venturadata_final)
ventura_preprocess$bc$yearbuilt

#pca on ventura
pca_ventura<-prcomp(venturadata_final,center=TRUE,scale.=TRUE)
percentvariance_ventura<-pca_ventura$sdev^2/sum(pca_ventura$sdev^2)*100

#correlation plot
corr_ventura<-cor(venturadata_final)
corrplot_ventura<-corrplot(corr_ventura[(1:10),(1:10)],order="hclust",tl.cex=0.8)

#splitting data
set.seed(598)
venturatrain<-createDataPartition(venturadata_final$regionidzip,p=0.8,list=FALSE)
ventura_trainingdata<-venturadata_final[venturatrain,]
ventura_testingdata<-venturadata_final[-venturatrain,]
ventura_predictor_trainingdata<-ventura_trainingdata[,-12]
ventura_response_trainingdata<-ventura_trainingdata[,12]
ventura_predictor_testingdata<-ventura_testingdata[,-12]
ventura_response_testingdata<-ventura_testingdata[,12]

#writing new csv for filtered data
write.csv(filtered_data,"filtered_data.csv")
write.csv(LAdata,"LA_data.csv")
write.csv(orangedata,"orangedata.csv")
write.csv(venturadata,"venturadata.csv")

#lm for ventura

lmventura<-lm(price~.,data=ventura_trainingdata)
summary(lmventura)
olmventura<-predict(lmventura,ventura_predictor_testingdata)
olmventuravalues<-data.frame(obs=ventura_response_testingdata,pred=olmventura)
defaultSummary(olmventuravalues)

#ventura high and low

quantile(venturadata_final$price, probs = c(0.8,0.85,0.9,0.95))
quantile(venturadata_final$price, probs = c(0,0.50,0.75,1))

#finding rows with house price>5000000
ventura_rows<-which(venturadata_final$taxvaluedollarcnt>500000)
venturadata_high<-venturadata_final[ventura_rows,] 
venturadata_low<-venturadata_final[-ventura_rows,]
venturadata_low<-venturadata_low[,-12]
set.seed(598)
venturatrain_low<-createDataPartition(venturadata_low$bathroomcnt,p=0.8,list=FALSE)
ventura_trainingdata_low<-venturadata_low[venturatrain_low,]
ventura_testingdata_low<-venturadata_low[-venturatrain_low,]
ventura_predictor_trainingdata_low<-ventura_trainingdata_low[,-13]
ventura_response_trainingdata_low<-ventura_trainingdata_low[,13]
ventura_predictor_testingdata_low<-ventura_testingdata_low[,-13]
ventura_response_testingdata_low<-ventura_testingdata_low[,13]
lmventura_low<-lm(price~.,data=ventura_trainingdata_low)
summary(lmventura_low)
olmventura_low<-predict(lmventura_low,ventura_predictor_testingdata_low)
olmventuravalues_low<-data.frame(obs=ventura_response_testingdata_low,pred=olmventura_low)
defaultSummary(olmventuravalues_low)

#lm for ventura with cv
ctrl<-trainControl(method="cv",number=10)
set.seed(576)
lmfitventura<-train(x=ventura_predictor_trainingdata,y=ventura_response_trainingdata,method="lm",trControl =ctrl)
olmfitventura<-predict(lmfitventura,ventura_predictor_testingdata)
olmfitventuravalues<-data.frame(obs=ventura_response_testingdata,pred=olmfitventura)
defaultSummary(olmfitventuravalues)
resid(lmfitventura)

#rlm for ventura
set.seed(576)
rlmpca_ventura<-train(ventura_predictor_trainingdata, ventura_response_trainingdata,
                      method="rlm",preProcess="pca",trControl=ctrl)
rlmpcapred_ventura<-predict(rlmpca_ventura,ventura_predictor_testingdata)
rlmpcadataset_ventura<-data.frame(obs=ventura_response_testingdata,pred=rlmpcapred_ventura)
defaultSummary(rlmpcadataset_ventura)

#pcr for ventura
pcr_ventura<-pcr()

#pls for ventura
#removing variables architecturalstyletypeid, buildingclasstypeid, fips
plstune_ventura<-train(ventura_predictor_trainingdata[,-c(3,7,9)],ventura_response_trainingdata,method="pls",
                       tuneLength=15,trControl=ctrl, preProc=c("center","scale"))

#ridgemodel for ventura
ridgemodel_ventura<-enet(x=as.matrix(ventura_predictor_trainingdata),y=ventura_response_trainingdata,lambda=0.001)
ridgepred_ventura<-predict(ridgemodel_ventura,newx=as.matrix(ventura_predictor_trainingdata),type="fit",
                           s=1,mode="fraction")

#lasso model for ventura
lassomodel_ventura<-enet(x=as.matrix(ventura_predictor_trainingdata),y=ventura_response_trainingdata,lambda=0.01,normalize=TRUE)
lassopred_ventura<-predict(lassomodel_ventura,newx=as.matrix(ventura_predictor_trainingdata),type="fit",
                           s=0.1,mode="fraction")
lassogrid_ventura<-expand.grid(.lambda=c(0,0.01,0.1),.fraction=seq(0.05,1,length=20))
set.seed(576)
lassotune_ventura<-train(ventura_predictor_trainingdata,method="enet",
                         tuneGrid=lassogrid_ventura, trControl)

#ORANGE

orangedata_predictor<-orangedata[,-25]
orangedata_response<-orangedata[,25]

#finding columns with zero variance
zerovar_orange<-nearZeroVar(orangedata_predictor)

#removing columns with zero variance
orangedata_nozerovar<-orangedata[,-zerovar_orange]

#finding categorical data in ventura
str(orangedata_nozerovar)

#removing factor data
orangedata_nofactor<-orangedata_nozerovar[,-c(12,14)]
orangedata_nofactor<-orangedata_nofactor[,-14]

#correlation of ventura_data
corr_orange<-cor(orangedata_nofactor)
cutoff_orange<-findCorrelation(corr_orange,cutoff=0.8)
orangedata_final<-orangedata_nofactor[,-cutoff_orange]
orangedata_final$taxamount<-orangedata_nofactor$taxamount #adding taxamount
orangedata_final<-orangedata_final[,-c(1,11)] #removing unwanted columns(parcel id, poolcnt(too many zeros))

#removing columns with high correlation
orangedata_final<-orangedata_final[,-12] 
orangedata_final<-orangedata_final[,-13] 
orangedata_final<-orangedata_final[,-c(1,4,8,9,13)]
orangedata_final$structuredollarvaluecnt<-orangedata$structuretaxvaluedollarcnt
orangedata_final$landtaxvaluedollarcnt<-orangedata$landtaxvaluedollarcnt
orangedata_final$taxamount<-orangedata_nofactor$taxamount
orangedata_final$price<-orangedata_response
orange_preprocess<-preProcess(orangedata_final,method=c("BoxCox","center","scale","pca"),
                              thresh=0.95,freqCut=20,uniqueCut=10,cutoff=0.8,fudge=0.2,numUnique=30)
orange_preprocessresults<-predict(orange_preprocess,orangedata_final)

##splitting data
set.seed(598)
orangetrain<-createDataPartition(orangedata_final$regionidzip,p=0.8,list=FALSE)
orange_trainingdata<-orangedata_final[orangetrain,]
orange_testingdata<-orangedata_final[-orangetrain,]
orange_predictor_trainingdata<-orange_trainingdata[,-11]
orange_response_trainingdata<-orange_trainingdata[,11]
orange_predictor_testingdata<-orange_testingdata[,-11]
orange_response_testingdata<-orange_testingdata[,11]

#lm for orange
lmorange<-lm(price~.,data=orange_trainingdata)
summary(lmorange)
olmorange<-predict(lmorange,orange_predictor_testingdata)
olmorangevalues<-data.frame(obs=orange_response_testingdata,pred=olmorange)
defaultSummary(olmorangevalues)
quantile(orangedata_final$price, probs = c(0.8,0.85,0.9,0.95))
quantile(orangedata_final$price, probs = c(0,0.50,0.75,1))

#finding rows with house price>5000000
orange_rows<-which(orangedata_final$price>500000)
orangedata_high<-orangedata_final[orange_rows,] 
orangedata_low<-orangedata_final[-orange_rows,]
orangedata_low<-orangedata_low[,-2]
set.seed(598)
orangetrain_low<-createDataPartition(orangedata_low$bathroomcnt,p=0.8,list=FALSE)
orange_trainingdata_low<-orangedata_low[orangetrain_low,]
orange_testingdata_low<-orangedata_low[-orangetrain_low,]
orange_predictor_trainingdata_low<-orange_trainingdata_low[,-11]
orange_response_trainingdata_low<-orange_trainingdata_low[,11]
orange_predictor_testingdata_low<-orange_testingdata_low[,-11]
orange_response_testingdata_low<-orange_testingdata_low[,11]
lmorange_low<-lm(price~.,data=orange_trainingdata_low)
summary(lmorange_low)
olmorange_low<-predict(lmorange_low,orange_predictor_testingdata_low)
olmorangevalues_low<-data.frame(obs=orange_response_testingdata_low,pred=olmorange_low)
defaultSummary(olmorangevalues_low)

#LOS ANGELES

LAdata_predictor<-LAdata[,-25]
LAdata_response<-LAdata[,25]

#finding columns with zero variance
zerovar_LA<-nearZeroVar(LAdata_predictor)

#removing columns with zero variance
LAdata_nozerovar<-LAdata[,-zerovar_LA]

#finding categorical data in LA
str(LAdata_nozerovar)

#removing factor data
LAdata_nofactor<-LAdata_nozerovar[,-c(12,14)]
LAdata_nofactor<-LAdata_nofactor[,-10] 
LAdata_nofactor<-LAdata_nofactor[,-14] # removing regionidneighbourhood(too many nas)
str(LAdata_nofactor)
#correlation of ventura_data
corr_LA<-cor(LAdata_nofactor)
cutoff_LA<-findCorrelation(corr_LA,cutoff=0.8)
LAdata_final<-LAdata_nofactor[,-cutoff_LA]
LAdata_final$taxamount<-LAdata_nofactor$taxamount #adding taxamount

#removing columns with high correlation
LAdata_final<-LAdata_final[,-6] 
LAdata_final<-LAdata_final[,-12] 
#removing low coerrelation columns
LAdata_final<-LAdata_final[,-c(6,8)]
LAdata_final<-LAdata_final[,-c(1,2,5,6,9,10,11,13)]
LAdata_final$structuredollarvaluecnt<-LAdata$structuretaxvaluedollarcnt
LAdata_final$landtaxvaluedollarcnt<-LAdata$landtaxvaluedollarcnt
LAdata_final$taxamount<-LAdata_nofactor$taxamount
LAdata_final$price<-LAdata_response
LA_preprocess<-preProcess(LAdata_final,method=c("BoxCox","center","scale","pca"),
                          thresh=0.95,freqCut=20,uniqueCut=10,cutoff=0.8,fudge=0.2,numUnique=30)
LA_preprocessresults<-predict(LA_preprocess,orangedata_final)

##splitting data
set.seed(598)
LAtrain<-createDataPartition(LAdata_final$bathroomcnt,p=0.8,list=FALSE)
LA_trainingdata<-LAdata_final[LAtrain,]
LA_testingdata<-LAdata_final[-LAtrain,]
LA_predictor_trainingdata<-LA_trainingdata[,-7]
LA_response_trainingdata<-LA_trainingdata[,7]
LA_predictor_testingdata<-LA_testingdata[,-7]
LA_response_testingdata<-LA_testingdata[,7]

#lm for LA
lmLA<-lm(price~.,data=LA_trainingdata)
summary(lmLA)
olmLA<-predict(lmLA,LA_predictor_testingdata)
olmLAvalues<-data.frame(obs=LA_response_testingdata,pred=olmLA)
defaultSummary(olmLAvalues)

#max and min values of price 
max_LA<-max(LAdata_final$price)
min_LA<-min(LAdata_final$price)
quantile(LAdata_final$price, probs = c(0.8,0.85,0.9,0.95))

#finding rows with house price> 1M
LA_rows<-which(LAdata_final$price>500000)
LAdata_high<-LAdata_final[LA_rows,] 
LAdata_low<-LAdata_final[-LA_rows,]

#LA low


LAdata_low<-LAdata_low[,-1]
LAdata_low<-LAdata_low[,-11]
set.seed(598)
LAtrain_low<-createDataPartition(LAdata_low$bathroomcnt,p=0.8,list=FALSE)
LA_trainingdata_low<-LAdata_low[LAtrain_low,]
LA_testingdata_low<-LAdata_low[-LAtrain_low,]
LA_predictor_trainingdata_low<-LA_trainingdata_low[,-14]
LA_response_trainingdata_low<-LA_trainingdata_low[,14]
LA_predictor_testingdata_low<-LA_testingdata_low[,-14]
LA_response_testingdata_low<-LA_testingdata_low[,14]
lmLA_low<-lm(price~.,data=LA_trainingdata_low)
summary(lmLA_low)
olmLA_low<-predict(lmLA_low,LA_predictor_testingdata_low)
olmLAvalues_low<-data.frame(obs=LA_response_testingdata_low,pred=olmLA_low)
defaultSummary(olmLAvalues_low)

#pls for ventura 
plstune_ventura<-train(ventura_predictor_trainingdata_low,ventura_response_trainingdata_low,
                       method="pls",tuneLength=13,trControl=ctrl,preProc=c("center","scale"))
testpls_ventura<-data.frame(obs=ventura_response_testingdata_low,pred=predict(plstune_ventura,ventura_predictor_testingdata_low))
defaultSummary(testpls_ventura)

#lm for ventura
lmfiltered_ventura<-train(x=ventura_predictor_trainingdata_low,y=ventura_response_trainingdata_low,method="lm",
                          trControl=ctrl,preProc = c("center", "scale"),na.action=na.omit)
testlm_ventura<-data.frame(obs=ventura_response_testingdata_low,pred=predict(lmfiltered_ventura,ventura_predictor_testingdata_low))
defaultSummary(testlm_ventura)

#ridge for ventura
ridgegrid_ventura<-data.frame(.lambda=0.1)
ridgetrain_ventura<-train(ventura_predictor_trainingdata_low,ventura_response_trainingdata_low,
                          method="ridge",tuneGrid=ridgegrid_ventura,trControl=ctrl,
                          preProc=c("center","scale"))
ridgetrain_ventura$results
testridge_ventura<-data.frame(obs=ventura_response_testingdata_low,pred=predict(ridgetrain_ventura,ventura_predictor_testingdata_low))
defaultSummary(testridge_ventura)

#enet for ventura
enetGrid_ventura <- expand.grid(lambda = c(0, 0.01, .1), fraction = seq(.05, 1, length = 20))
set.seed(100)
enetTune_ventura <- train(x = ventura_predictor_trainingdata_low, y = ventura_response_trainingdata_low,
                          method = "enet",
                          tuneGrid = enetGrid_ventura,
                          trControl = ctrl,
                          preProc = c("center", "scale"))
testenet_ventura<-data.frame(obs=ventura_response_testingdata_low,pred=predict(enetTune_ventura,ventura_predictor_testingdata_low))
defaultSummary(testenet_ventura)

#rlm for ventura
rlmPCA_ventura <- train(x =  ventura_predictor_trainingdata_low, y = ventura_response_trainingdata_low, method = "rlm", preProcess = "pca", trControl = ctrl)
testResultsPCA_ventura <- data.frame(obs = ventura_response_testingdata_low, pred = predict(rlmPCA_ventura, ventura_predictor_testingdata_low))
defaultSummary(testResultsPCA_ventura)

#nnet without train
nnetfit_ventura<-nnet(x=ventura_predictor_trainingdata_low,y = ventura_response_trainingdata_low,size=5,decay=0.01,
                      linout=TRUE,trace=FALSE,maxit=500,MaxNWts=5*(ncol(ventura_predictor_trainingdata_low)+1)+6)

#nnet with train
nnetmodel_ventura<-train(x =  ventura_predictor_trainingdata_low, y = ventura_response_trainingdata_low,metho="nnet",preProc=c("center","scale"),linout=TRUE,trace=FALSE,
                         MaxNwts=10*(ncol(ventura_predictor_trainingdata_low)+1)+11,maxit=500)
nnetpre_ventura<- data.frame(obs = ventura_response_testingdata_low, pred = predict(nnetmodel_ventura, ventura_predictor_testingdata_low))
defaultSummary(nnetpre_ventura)

#pls for orange 
plstune_orange<-train(orange_predictor_trainingdata_low,orange_response_trainingdata_low,
                      method="pls",tuneLength=13,trControl=ctrl,preProc=c("center","scale"))
testpls_orange<-data.frame(obs=orange_response_testingdata_low,pred=predict(plstune_orange,orange_predictor_testingdata_low))
defaultSummary(testpls_orange)


#lm for orange
lmfiltered_orange<-train(x=orange_predictor_trainingdata_low,y=orange_response_trainingdata_low,method="lm",
                         trControl=ctrl)
testlm_orange<-data.frame(obs=orange_response_testingdata_low,pred=predict(lmfiltered_orange,orange_predictor_testingdata_low))
defaultSummary(testlm_orange)

#ridge for orange
ridgegrid_orange<-data.frame(.lambda=seq(0, 0.1, length = 15))
ridgetrain_orange<-train(orange_predictor_trainingdata_low,orange_response_trainingdata_low,
                         method="ridge",tuneGrid=ridgegrid_orange,trControl=ctrl,
                         preProc=c("center","scale"))
ridgetrain_orange$results
testridge_orange<-data.frame(obs=orange_response_testingdata_low,pred=predict(ridgetrain_orange,orange_predictor_testingdata_low))
defaultSummary(testridge_orange)

#enet for orange
enetGrid_orange <- expand.grid(lambda = c(0, 0.01, .1), fraction = seq(.05, 1, length = 20))
set.seed(100)
enetTune_orange <- train(x = orange_predictor_trainingdata_low, y = orange_response_trainingdata_low,
                         method = "enet",
                         tuneGrid = enetGrid_orange,
                         trControl = ctrl,
                         preProc = c("center", "scale"))
testenet_orange<-data.frame(obs=orange_response_testingdata_low,pred=predict(enetTune_orange,orange_predictor_testingdata_low))
defaultSummary(testenet_orange)

#rlm for orange
rlmPCA_orange <- train(x =  orange_predictor_trainingdata_low, y = orange_response_trainingdata_low, method = "rlm", preProcess = "pca", trControl = ctrl)
testResultsPCA_orange <- data.frame(obs = orange_response_testingdata_low, pred = predict(rlmPCA_orange, orange_predictor_testingdata_low))
defaultSummary(testResultsPCA_orange)


#pls for LA 
plstune_LA<-train(LA_predictor_trainingdata_low,LA_response_trainingdata_low,
                  method="pls",tuneLength=13,trControl=ctrl,preProc=c("center","scale"))
testpls_LA<-data.frame(obs=LA_response_testingdata_low,pred=predict(plstune_LA,LA_predictor_testingdata_low))
defaultSummary(testpls_LA)


#lm for LA
lmfiltered_LA<-train(x=LA_predictor_trainingdata_low,y=LA_response_trainingdata_low,method="lm",
                     trControl=ctrl,preProc=c("center","scale"))
testlm_LA<-data.frame(obs=LA_response_testingdata_low,pred=predict(lmfiltered_LA,LA_predictor_testingdata_low))
defaultSummary(testlm_LA)

#ridge for LA
ridgegrid_LA<-data.frame(.lambda=seq(0, 0.1, length = 15))
ridgetrain_LA<-train(LA_predictor_trainingdata_low,LA_response_trainingdata_low,
                     method="ridge",tuneGrid=ridgegrid_LA,trControl=ctrl,
                     preProc=c("center","scale"))
ridgetrain_LA$results
testridge_LA<-data.frame(obs=LA_response_testingdata_low,pred=predict(ridgetrain_LA,LA_predictor_testingdata_low))
defaultSummary(testridge_LA)

#enet for LA
enetGrid_LA <- expand.grid(lambda = c(0, 0.01, .1), fraction = seq(.05, 1, length = 20))
set.seed(100)
enetTune_LA <- train(x = LA_predictor_trainingdata_low, y = LA_response_trainingdata_low,
                     method = "enet",
                     tuneGrid = enetGrid_LA,
                     trControl = ctrl,
                     preProc = c("center", "scale"))
testenet_LA<-data.frame(obs=LA_response_testingdata_low,pred=predict(enetTune_LA,LA_predictor_testingdata_low))
defaultSummary(testenet_LA)

#rlm for LA
rlmPCA_LA <- train(x =  LA_predictor_trainingdata_low, y = LA_response_trainingdata_low, method = "rlm", preProcess = "pca", trControl = ctrl)
testResultsPCA_LA <- data.frame(obs = LA_response_testingdata_low, pred = predict(rlmPCA_LA, LA_predictor_testingdata_low))
defaultSummary(testResultsPCA_LA)

## Boosted Tree and Random forest code
ventura_newdata <- read.csv("venturadata.csv", quote = "") 
rows<-which(ventura_data$taxvaluedollarcnt<=500000)
ventura_data<-ventura_data[rows,]
ventura_filtered <- ventura_data[,c(1,2,4:6,8,10,12,14,15,21:23,25)]
M <- cor(ventura_filtered)
set.seed(576)

#create partition
sample_ventura <- sample(1:nrow(ventura_filtered), size=0.8*nrow(ventura_filtered)) 
train_ventura <- ventura_filtered[sample_ventura,] 
test_ventura <- ventura_filtered[-sample_ventura,] 
test1 <- ventura_filtered[-sample_ventura, "taxvaluedollarcnt"]

#boosted tree regression
Boost_ventura <- gbm(taxvaluedollarcnt~.,data = train_ventura, distribution = "gaussian", n.trees = 1000)
summary(Boost_ventura)
yhat.boost <- predict(Boost_ventura, newdata = test_ventura, n.trees = 1000)
mean((yhat.boost-test_ventura)^2)
Boostpr <- round(postResample(pred = yhat.boost, obs = test1),2)
Boostpr

#random forest tree regression
RF_ventura <- randomForest(taxvaluedollarcnt~.,data = train_ventura, ntree = 500, importance= TRUE)
varImpPlot(RF_ventura, type = 1)
yhat.rf <- predict(RF_ventura, newdata = test_ventura)
RFpr <- round(postResample(pred = yhat.rf, obs = test1),2)
RFpr

#tune max depth by creating 10 fold cv
ctrl <- trainControl(method = "cv", number = 10)
rpartTune <- train(taxvaluedollarcnt~.,data = train_ventura, method = "rpart2",tuneLength = 10, trControl = ctrl)
rpartTune

##Orange county Random forest code
orange_newdata <- read.csv("orangedata.csv", quote = "") 
row2<-which(orange_newdata$taxvaluedollarcnt<=500000)
orange_newdata<-orange_newdata[row2,]
orange_filtered <- orange_newdata[,c(1,2,4:6,8,10,12,14,15,21:23,25)]
N <- cor(orange_filtered)
corrplot(N, method = "shade")
set.seed(576)

#create partition
sample_orange <- sample(1:nrow(orange_filtered), size=0.8*nrow(orange_filtered)) 
train_orange <- orange_filtered[sample_orange,] 
test_orange <- orange_filtered[-sample_orange,] 
test2 <- orange_filtered[-sample_orange, "taxvaluedollarcnt"]

#boosted regression
Boost_orange <- gbm(taxvaluedollarcnt~.,data = train_orange, distribution = "gaussian", n.trees = 1000)
summary(Boost_orange)
yhat.boost1 <- predict(Boost_orange, newdata = test_orange, n.trees = 1000)
mean((yhat.boost1-test_orange)^2)
Boostpr1 <- round(postResample(pred = yhat.boost1, obs = test2),2)
Boostpr1

#random forest tree regression
systime()
RF_orange <- randomForest(taxvaluedollarcnt~.,data = train_orange, ntree = 500, importance= TRUE)
varImpPlot(RF_orange, type = 1)
yhat.rf1 <- predict(RF_orange, newdata = test_orange)
RFpr1 <- round(postResample(pred = yhat.rf1, obs = test2),2)
RFpr1
systime()

#tune max depth by creating 10 fold cv
ctrl <- trainControl(method = "cv", number = 10)
rpartTune1 <- train(taxvaluedollarcnt~.,data = train_orange, method = "rpart2",tuneLength = 10, trControl = ctrl)
rpartTune1
FinalTree1 = rpartTune1$finalModel
rpartTree1 = as.party(FinalTree1)
dev.new()
plot(rpartTree1)

##LA county Random forest code
LA_newdata <- read.csv("LA_data1.csv", quote = "") 
row3 <- which(LA_newdata$taxvaluedollarcnt<=500000)
LA_newdata<-LA_newdata[row3,]
LA_filtered <- LA_newdata[,c(1,2,4:6,8,10,12,14,15,21:23,25)]
set.seed(576)

#create partition
sample_LA <- sample(1:nrow(LA_filtered), size=0.8*nrow(LA_filtered)) 
train_LA <- LA_filtered[sample_LA,] 
test_LA <- LA_filtered[-sample_LA,] 
test3 <- LA_filtered[-sample_LA, 14]

#boosted regression
Boost_LA <- gbm(taxvaluedollarcnt~.,data = train_LA, distribution = "gaussian", n.trees = 1000)
summary(Boost_LA)
yhat.boost2 <- predict(Boost_LA, newdata = test_LA, n.trees = 1000)
mean((yhat.boost2-test_LA)^2)
Boostpr2 <- round(postResample(pred = yhat.boost2, obs = test3),2)
Boostpr2

#Random forest regression
RF_LA <- randomForest(taxvaluedollarcnt~.,data = train_LA, ntree = 500, importance= TRUE)
varImpPlot(RF_LA, type = 1 )
yhat.rf2 <- predict(RF_LA, newdata = test_LA)
RFpr2 <- round(postResample(pred = yhat.rf2, obs = test3),2)
RFpr2

## Neural Network
data_train <- read.csv("LA_train.csv",quote="")
data_train$type <- 0
data_test <- read.csv("LA_test.csv",quote="")
data_test$type <- 1
df <- rbind(data_train, data_test)
df$X.buildingqualitytypeid. <- as.numeric(df$X.buildingqualitytypeid.)

# remove unwanted cols
df <- df[ -c(1:2,4,8,10,12,19,21,27) ]

# convert categorical cols to factors
cols <- c("X.airconditioningtypeid.",
          "X.heatingorsystemtypeid.","X.propertylandusetypeid.",
          "X.regionidcity.","X.regionidzip.")
df[cols] <- lapply(df[cols], factor)
df <- df[ -c(1,4:5,7,10:13)]

# create dummy variables for factors
class(df$X.buildingqualitytypeid.)

# remove variables that are factors
df_numeric <- df[, sapply(df, class) != "factor"]
df_numeric <- as.data.frame(sapply(df_numeric, as.numeric))
df_numeric <- subset(df_numeric, X.taxvaluedollarcnt. >=100000 & X.taxvaluedollarcnt. <= 400000)
df_numeric_train <- subset(df_numeric, type == 0)
df_numeric_test <- subset(df_numeric, type == 1)
input <- df_numeric_train[,c(1:8,10:11)]
test_input <- df_numeric_test[,c(1:8,10:11)]
scaled.input <- as.data.frame(scale(df_numeric_train[,c(1:8,10:11)]))
scaled.input_test <- as.data.frame(scale(df_numeric_test[c(1:8,10:11)]))
response <- df_numeric_train$X.taxvaluedollarcnt.
test_response <- df_numeric_test$X.taxvaluedollarcnt.

# , tuneGrid = nnGrid
nnetModel = train(x=input, y=response, method="nnet", preProc=c("center", "scale"), linout=TRUE, trace=FALSE,
                  MaxNWts=10 * (ncol(scaled.input)+1) + 10 + 1, maxit=500)
scaled.input_test$X.garagecarcnt. <- -0.0007564184
0.0007564184
nnetModel

# NNET
set.seed(50)
nnet_fit <- nnet(scaled.input, response, size = 5, decay = 0.01,
                 linout = TRUE, trace = FALSE, maxit = 500,
                 MaxNWts = 5 * (ncol(scaled.input) + 1) +5 +1)

pred_nnet <- predict(nnet_fit, scaled.input_test)
nnetPR = postResample(pred=pred_nnet, obs=test_response)
nnetPR
nnetPR[1]
nnetPR[2]
nnetPR[3]

# AVNNET
avNNetModel = avNNet(input,response,size = 5,linout=TRUE,trace=FALSE, maxit=500)
avNNetModel

# Lets see what variables are most important: 
varImp(avNNetModel)

avNNetPred = predict(avNNetModel, newdata=test_input)
avNNetPR = postResample(pred=avNNetPred, obs=test_response)
avNNetPR

## KNN
names <- c("Fake")
seeds <- c(0)
trainsize <- c(0)
k_values <- c(0)
rmses <- c(0)
r2s <- c(0)

Sys.time()

for (xx in c(101:120)){
  
  LA_Train_Pred <- read.csv("LA_Training_Predictor.csv",quote="")
  LA_Train_Resp <- read.csv("LA_Training_Response.csv",quote="")
  LA_Test_Pred <- read.csv("LA_Testing_Predictor.csv",quote="")
  LA_Test_Resp <- read.csv("LA_Testing_Response.csv",quote="")
  
  OC_Train_Pred <- read.csv("OC_Training_Predictor.csv",quote="")
  OC_Train_Resp <- read.csv("OC_Training_Response.csv",quote="")
  OC_Test_Pred <- read.csv("OC_Testing_Predictor.csv",quote="")
  OC_Test_Resp <- read.csv("OC_Testing_Response.csv",quote="")
  
  VC_Train_Pred <- read.csv("VC_Training_Predictor.csv",quote="")
  VC_Train_Resp <- read.csv("VC_Training_Response.csv",quote="")
  VC_Test_Pred <- read.csv("VC_Testing_Predictor.csv",quote="")
  VC_Test_Resp <- read.csv("VC_Testing_Response.csv",quote="")
  
  
  LA_Train_Pred <- LA_Train_Pred[, -c(1:4,7, 10:11,15, 17)]
  LA_Test_Pred <- LA_Test_Pred[, -c(1:4,7, 10:11, 15, 17)]
  
  OC_Train_Pred <- OC_Train_Pred[, -c(1:4,7, 15, 17)]
  OC_Test_Pred <- OC_Test_Pred[, -c(1:4,7, 15, 17)]
  
  VC_Train_Pred <- VC_Train_Pred[, -c(1:4,7, 15, 17)]
  VC_Test_Pred <- VC_Test_Pred[, -c(1:4,7, 15, 17)]
  
  
  #######################################
  # LA COUNTY
  
  set.seed(xx)
  
  knn_Grid <- expand.grid(k = c(9))
  
  rows <- complete.cases(LA_Train_Pred)
  LA_Train_Pred <- LA_Train_Pred[rows,]
  LA_Train_Resp <- LA_Train_Resp$X.X.x..[rows]
  
  rows <- complete.cases(LA_Test_Pred)
  LA_Test_Pred <- LA_Test_Pred[rows,]
  LA_Test_Resp <- LA_Test_Resp$x[rows]
  
  rows <- which(LA_Train_Resp <= 500000)
  LA_Train_Resp <- LA_Train_Resp[rows]
  LA_Train_Pred <- LA_Train_Pred[rows,]
  
  rows <- which(LA_Test_Resp <= 500000)
  LA_Test_Resp <- LA_Test_Resp[rows]
  LA_Test_Pred <- LA_Test_Pred[rows,]
  
  samplerows <- sample(nrow(LA_Train_Pred), floor(nrow(LA_Train_Pred)/100))
  LA_Train_Pred <- LA_Train_Pred[samplerows, ]
  LA_Train_Resp <- LA_Train_Resp[samplerows]
  
  samplerows <- sample(nrow(LA_Test_Pred), floor(nrow(LA_Test_Pred)/200))
  LA_Test_Pred <- LA_Test_Pred[samplerows, ]
  LA_Test_Resp <- LA_Test_Resp[samplerows]
  
  knnModel <- train(
    x = LA_Train_Pred, y = LA_Train_Resp, method = "knn",
    preProcess = c("center","scale"),
    tuneGrid = knn_Grid
  )
  knnModel
  
  colnames(LA_Test_Pred) <- colnames(LA_Train_Pred)
  
  knnPred = predict(knnModel, newdata=LA_Test_Pred)
  knnPR = postResample(pred=knnPred, obs=LA_Test_Resp)
  knnPR
  
  names <- c(names, "LA")
  seeds <- c(seeds, xx)
  trainsize <- c(trainsize, nrow(LA_Train_Pred))
  k_values <- c(k_values, knnModel$bestTune$k)
  rmses <- c(rmses, knnPR[1])
  r2s <- c(r2s, knnPR[2])
  
  #######################################
  # Orange COUNTY
  
  set.seed(xx)
  
  knn_Grid <- expand.grid(k = c(7))
  
  rows <- complete.cases(OC_Train_Pred)
  OC_Train_Pred <- OC_Train_Pred[rows,]
  OC_Train_Resp <- OC_Train_Resp$x[rows]
  
  rows <- complete.cases(OC_Test_Pred)
  OC_Test_Pred <- OC_Test_Pred[rows,]
  OC_Test_Resp <- OC_Test_Resp$x[rows]
  
  rows <- which(OC_Train_Resp <= 500000)
  OC_Train_Resp <- OC_Train_Resp[rows]
  OC_Train_Pred <- OC_Train_Pred[rows,]
  
  rows <- which(OC_Test_Resp <= 500000)
  OC_Test_Resp <- OC_Test_Resp[rows]
  OC_Test_Pred <- OC_Test_Pred[rows,]
  
  samplerows <- sample(nrow(OC_Train_Pred), floor(nrow(OC_Train_Pred)/20))
  OC_Train_Pred <- OC_Train_Pred[samplerows, ]
  OC_Train_Resp <- OC_Train_Resp[samplerows]
  
  samplerows <- sample(nrow(OC_Test_Pred), floor(nrow(OC_Test_Pred)/30))
  OC_Test_Pred <- OC_Test_Pred[samplerows, ]
  OC_Test_Resp <- OC_Test_Resp[samplerows]
  
  knnModel <- train(
    x = OC_Train_Pred, y = OC_Train_Resp, method = "knn",
    preProcess = c("center","scale"),
    tuneGrid = knn_Grid
  )
  knnModel
  
  knnPred = predict(knnModel, newdata=OC_Test_Pred)
  knnPR = postResample(pred=knnPred, obs=OC_Test_Resp)
  knnPR
  
  names <- c(names, "OC")
  seeds <- c(seeds, xx)
  trainsize <- c(trainsize, nrow(OC_Train_Pred))
  k_values <- c(k_values, knnModel$bestTune$k)
  rmses <- c(rmses, knnPR[1])
  r2s <- c(r2s, knnPR[2])
  
  #######################################
  # Ventura COUNTY
  set.seed(xx)
  
  knn_Grid <- expand.grid(k = c(11))
  
  rows <- complete.cases(VC_Train_Pred)
  VC_Train_Pred <- VC_Train_Pred[rows,]
  VC_Train_Resp <- VC_Train_Resp$x[rows]
  
  rows <- complete.cases(VC_Test_Pred)
  VC_Test_Pred <- VC_Test_Pred[rows,]
  VC_Test_Resp <- VC_Test_Resp$x[rows]
  
  rows <- which(VC_Train_Resp <= 500000)
  VC_Train_Resp <- VC_Train_Resp[rows]
  VC_Train_Pred <- VC_Train_Pred[rows,]
  
  rows <- which(VC_Test_Resp <= 500000)
  VC_Test_Resp <- VC_Test_Resp[rows]
  VC_Test_Pred <- VC_Test_Pred[rows,]
  
  samplerows <- sample(nrow(VC_Train_Pred), floor(nrow(VC_Train_Pred)/10))
  VC_Train_Pred <- VC_Train_Pred[samplerows, ]
  VC_Train_Resp <- VC_Train_Resp[samplerows]
  
  samplerows <- sample(nrow(VC_Test_Pred), floor(nrow(VC_Test_Pred)/15))
  VC_Test_Pred <- VC_Test_Pred[samplerows, ]
  VC_Test_Resp <- VC_Test_Resp[samplerows]
  
  knnModel <- train(
    x = VC_Train_Pred, y = VC_Train_Resp, method = "knn",
    preProcess = c("center","scale"),
    tuneGrid = knn_Grid
  )
  knnModel
  
  
  knnPred = predict(knnModel, newdata=VC_Test_Pred)
  knnPR = postResample(pred=knnPred, obs=VC_Test_Resp)
  knnPR
  
  names <- c(names, "VC")
  seeds <- c(seeds, xx)
  trainsize <- c(trainsize, nrow(VC_Train_Pred))
  k_values <- c(k_values, knnModel$bestTune$k)
  rmses <- c(rmses, knnPR[1])
  r2s <- c(r2s, knnPR[2])
  
}

Sys.time()

res = data.frame( name=names, seed=seeds, trainsize = trainsize, k_value= k_values, rmse = rmses, r2 = r2s )

res = res[ order(res$rmse ), ]
print( "Final Results" ) 
print( res )

write.csv(res, file="Results_20seeds_set1.csv", row.names=FALSE)
















