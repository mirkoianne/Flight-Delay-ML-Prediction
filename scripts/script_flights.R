#https://www.kaggle.com/datasets/threnjen/2019-airline-delays-and-cancellations?select=test.csv

data<-read.csv("flight_delay.csv", sep=",", na.strings=c("NA","NaN","NULL"))
str(data)

#Formatting variables
data$MONTH <- as.factor(data$MONTH)
data$DAY_OF_WEEK <- as.factor(data$DAY_OF_WEEK)
data$DEP_TIME_BLK <- as.factor(data$DEP_TIME_BLK)
data$DEP_DEL <- ifelse(data$DEP_DEL15 == 0, "On_time", "Delay") 
# created a new variable to replace the variabile about delays of at least 15 min. 
# DEP_DEL15 was encoded as 0 or 1, but for classification analysis and, above all, for the readability of the results (e.g., confusion matrix, graphs), it is useful to have an explicit, categorical variable that clearly represents the predicted class.
data$DEP_DEL <- as.factor(data$DEP_DEL)
summary(data)

# Excluded variables: distance_group, carrier_name, departing_airport, and previous_airport.
data2 <- data[,-c(5,9,18,21)]

# Target Distribution
table(data2$DEP_DEL)/nrow(data2)
# The event is not rare, so there is no need to balance the dataset.
# The target distribution is realistic, so the dataset was not
# previously balanced and we won't need to adjust the posteriors.
# I suggest sensitivity as the tuning metric for the models, to minimize delayed flights
# predicted as on-time, so airlines don't have to refund tickets.

# Missing values
library(funModeling)
status=df_status(data2, print_results = F)
status # non ci sono NA

#Correlations
#cor_mat <- cor(data2[,-c(1,2,4,23)], use='complete.obs')
#cor_mat

# Creating training, validation, and score datasets for step 4
library(caret)
set.seed(1234)
splitA <- createDataPartition(y=data2$DEP_DEL, p = 0.10, list = FALSE)
dati_storici <- data2[-splitA,]
score_data <- data2[splitA,]
score_cov <- score_data[,-23]
splitB <- createDataPartition(y=dati_storici$DEP_DEL, p = 0.67, list = FALSE)
ds.training <- dati_storici[splitB,]
ds.validation <- dati_storici[-splitB,]

# STEP 1: Model estimation and overfitting evaluation
# ESTIMATE MODELS FIRST AND THEN CALCULATE OVERFITTING, OTHERWISE VARIABLES ARE READ INCORRECTLY

# Decision Tree with caret tuning the best cp
set.seed(1)
Ctrl_tree <- trainControl(method = "cv" , number=10, summaryFunction = twoClassSummary,
                          classProbs = TRUE)
Tree <- train(DEP_DEL~.-DEP_DEL15, data = ds.training, method = "rpart", 
              tuneLength = 15, trControl = Ctrl_tree, minsplit = 5)
Tree
confusionMatrix(Tree)
validation_pred_prob_tree <- predict(Tree, ds.validation, type = "prob")     # calculating validation ASE
#head(validation_pred_prob_tree)
#ds.validation$diff_pred_tree <- (1-ds.validation$DEP_DEL15) - validation_pred_prob_tree[,1]
#summary(ds.validation$diff_pred_tree)
ASE_validation_tree <- (sum(( (1-ds.validation$DEP_DEL15) - validation_pred_prob_tree[,1] )^2)
                        / nrow(ds.validation))
training_pred_prob_tree <- predict(Tree, ds.training, type = "prob")     # calculating training ASE
ASE_training_tree <- (sum(( (1-ds.training$DEP_DEL15) - training_pred_prob_tree[,1] )^2)
                      / nrow(ds.training))
overfitting_tree = (ASE_training_tree - ASE_validation_tree) / ASE_training_tree
overfitting_tree # OK

#Random forest
set.seed(7)
Ctrl_rf <- trainControl(method="cv", number=10, search="grid",
                        summaryFunction = twoClassSummary, classProbs = TRUE)
tunegrid <- expand.grid(.mtry=4)          #or gap (1:7), centered in [sqrt(22)]
RF <- train(DEP_DEL~.-DEP_DEL15, data=ds.training, method="rf", tuneGrid=tunegrid,
            ntree=250, trControl=Ctrl_rf)
RF
#plot(RF)
confusionMatrix(RF)
validation_pred_prob_rf <- predict(RF, ds.validation, type = "prob")   # calculating validation ASE
#head(validation_pred_prob_rf)
ASE_validation_rf <- (sum(( (1-ds.validation$DEP_DEL15) - validation_pred_prob_rf[,1] )^2)
                         / nrow(ds.validation))
training_pred_prob_rf <- predict(RF, ds.training, type = "prob")       # calculating training ASE
ASE_training_rf <- (sum(( (1-ds.training$DEP_DEL15) - training_pred_prob_rf[,1] )^2)
                       / nrow(ds.training))
overfitting_rf = (ASE_training_rf - ASE_validation_rf) / ASE_training_rf
overfitting_rf # Random Forest OVERFITTS!

#Model selection with Random Forest
VarImportance <- varImp(RF)
plot(VarImportance)
VImP=as.data.frame(VarImportance$importance)
V=t(subset(VImP, Overall>20))
# Selected variables: SEGMENT_NUMBER, CONCURRENT_FLIGHTS, NUMBER_OF_SEATS, AIRPORT_FLIGHTS_MONTH
# AIRLINE_FLIGHTS_MONTH, AIRLINE_AIRPORT_FLIGHTS_MONTH, AVG_MONTHLY_PASS_AIRPORT, AVG_MONTHLY_PASS_AIRLINE
# FLT_ATTENDANTS_PER_PASS, GROUND_SERV_PER_PASS, PLANE_AGE, LATITUDE, LONGITUDE, PRCP, TMAX, AWND
Xselected_training=ds.training[,colnames(V)]
Xselected_validation=ds.validation[,colnames(V)]
ds.training.modelsel=cbind(Xselected_training, ds.training[,c(3,23)])
ds.validation.modelsel=cbind(Xselected_validation, ds.validation[,c(3,23)])

#Lasso
set.seed(555)
grid = expand.grid(.alpha=1,.lambda=seq(0, 1, by = 0.01))
Ctrl_lasso=trainControl(method="cv",number=10, summaryFunction=twoClassSummary,
                        classProbs=TRUE)
Lasso=train(DEP_DEL~.-DEP_DEL15,data=ds.training, method="glmnet", family="binomial",
            preProcess=c("scale","nzv"), trControl=Ctrl_lasso, tuneLength=5, tuneGrid=grid)
Lasso
plot(Lasso) # mostra graficamente quale sia il miglior lambda
coef(Lasso$finalModel, s=Lasso$bestTune$lambda)
confusionMatrix(Lasso)
validation_pred_prob_lasso <- predict(Lasso, ds.validation, type = "prob")   # calculating validation ASE
#head(validation_pred_prob_lasso)
ASE_validation_lasso <- (sum(( (1-ds.validation$DEP_DEL15) - validation_pred_prob_lasso[,1] )^2)
                        / nrow(ds.validation))
training_pred_prob_lasso <- predict(Lasso, ds.training, type = "prob")       # calculating training ASE
ASE_training_lasso <- (sum(( (1-ds.training$DEP_DEL15) - training_pred_prob_lasso[,1] )^2)
                      / nrow(ds.training))
overfitting_lasso = (ASE_training_lasso - ASE_validation_lasso) / ASE_training_lasso
overfitting_lasso # OK

#Logistic Regression
set.seed(447)
Control_glm=trainControl(method= "cv", number=10, classProbs = TRUE)
glm=train(DEP_DEL~.-DEP_DEL15, data=ds.training, method = "glm", preProcess=c("corr", "nzv" , "scale",
          "BoxCox"), trControl = Control_glm, tuneLength=5, trace=FALSE)
glm
validation_pred_prob_glm <- predict(glm, ds.validation, type = "prob")   # calculating validation ASE
#head(validation_pred_prob_glm)
ASE_validation_glm <- (sum(( (1-ds.validation$DEP_DEL15) - validation_pred_prob_glm[,1] )^2)
                        / nrow(ds.validation))
training_pred_prob_glm <- predict(glm, ds.training, type = "prob")       # calculating training ASE
ASE_training_glm <- (sum(( (1-ds.training$DEP_DEL15) - training_pred_prob_glm[,1] )^2)
                      / nrow(ds.training))
overfitting_glm = (ASE_training_glm - ASE_validation_glm) / ASE_training_glm
overfitting_glm # OK

#PLS
set.seed(815)
Control_pls=trainControl(method= "cv",number=10, summaryFunction=twoClassSummary, classProbs=TRUE)
pls=train(DEP_DEL~.-DEP_DEL15,data=ds.training.modelsel , method = "pls", preProcess=c("scale"),
          trControl = Control_pls, tuneLength=5)
pls
validation_pred_prob_pls <- predict(pls, ds.validation.modelsel, type = "prob") # calculating validation ASE
#head(validation_pred_prob_pls)
ASE_validation_pls <- (sum(( (1-ds.validation.modelsel$DEP_DEL15) - validation_pred_prob_pls[,1] )^2)
                         / nrow(ds.validation.modelsel))
training_pred_prob_pls <- predict(pls, ds.training.modelsel, type = "prob")     # calculating training ASE
ASE_training_pls <- (sum(( (1-ds.training.modelsel$DEP_DEL15) - training_pred_prob_pls[,1] )^2)
                       / nrow(ds.training.modelsel))
overfitting_pls = (ASE_training_pls - ASE_validation_pls) / ASE_training_pls
overfitting_pls # OK

#Naive Bayes
ds.training2 <- ds.training[ , !(colnames(ds.training) %in% "DEP_DEL15")]
#ds.training2 removes the DEP_DEL15 variable because it caused issues with the naivebayes code
library(klaR)
naive_all <- NaiveBayes(DEP_DEL~., data = ds.training2, usekernel = FALSE, laplace = 100,
                        na.action=na.exclude, preProcess=c("corr","nzv","BoxCox"))
naive_all
validation_pred_prob_nb <- predict(naive_all, ds.validation, type = "prob") # calculating validation ASE
head(validation_pred_prob_nb)
nb_posterior_validation <- validation_pred_prob_nb$posterior
head(nb_posterior_validation[,1])
ASE_validation_nb <- (sum(( (1-ds.validation$DEP_DEL15) - nb_posterior_validation[,1] )^2)
                       / nrow(ds.validation))
training_pred_prob_nb <- predict(naive_all, ds.training, type = "prob")     # calculating training ASE
nb_posterior_training <- training_pred_prob_nb$posterior
ASE_training_nb <- (sum(( (1-ds.training$DEP_DEL15) - nb_posterior_training[,1] )^2)
                     / nrow(ds.training))
overfitting_nb = (ASE_training_nb - ASE_validation_nb) / ASE_training_nb
overfitting_nb # OK
confusionMatrix(validation_pred_prob_nb$class, ds.validation$DEP_DEL)

#knn
set.seed(11)
Ctrl_knn <- trainControl(method = "cv", number=10, searc="grid", 
                       summaryFunction = twoClassSummary, classProbs = TRUE)
knn <- train(DEP_DEL~.-DEP_DEL15, data=ds.training.modelsel, method = "knn", tuneLength = 10,
                 preProcess = c("center", "scale", "corr","nzv"), trControl = Ctrl_knn)
knn
validation_pred_prob_knn <- predict(knn, ds.validation.modelsel, type = "prob") # calculating validation ASE
#head(validation_pred_prob_knn)
ASE_validation_knn <- (sum(( (1-ds.validation.modelsel$DEP_DEL15) - validation_pred_prob_knn[,1] )^2)
                       / nrow(ds.validation.modelsel))
training_pred_prob_knn <- predict(knn, ds.training.modelsel, type = "prob")     # calculating training ASE
ASE_training_knn <- (sum(( (1-ds.training.modelsel$DEP_DEL15) - training_pred_prob_knn[,1] )^2)
                     / nrow(ds.training.modelsel))
overfitting_knn = (ASE_training_knn - ASE_validation_knn) / ASE_training_knn
overfitting_knn # OK

#Neural Network
set.seed(40)
ctrl_nnet = trainControl(method="cv", number=10, search="grid",
                         summaryFunction = twoClassSummary, classProbs = TRUE)
metric <- 'ROC'
tunegrid <- expand.grid(size= (1:5), decay = .1)
nnet <- train(ds.training.modelsel[-c(17,18)], ds.training.modelsel$DEP_DEL,   #I use  x, y
                         method = "nnet", preProcess = c("scale","nzv","corr"),
                         tuneLength = 5, trControl=ctrl_nnet, tuneGrid=tunegrid, metric=metric,
                         trace = TRUE, #  to track interactions
                         maxit = 300) # Neural Network converges 
nnet
plot(nnet)
confusionMatrix(nnet)
validation_pred_prob_nnet <- predict(nnet, ds.validation, type = "prob")      # calculating validation ASE
#head(validation_pred_prob_nnet)
ASE_validation_nnet <- (sum(( (1-ds.validation$DEP_DEL15) - validation_pred_prob_nnet[,1] )^2)
                       / nrow(ds.validation))
training_pred_prob_nnet <- predict(nnet, ds.training, type = "prob")          # calculating training ASE
ASE_training_nnet <- (sum(( (1-ds.training$DEP_DEL15) - training_pred_prob_nnet[,1] )^2)
                     / nrow(ds.training))
overfitting_nnet = (ASE_training_nnet - ASE_validation_nnet) / ASE_training_nnet
overfitting_nnet # OK

#STEP 2: assessment
#ROC Curves
library(pROC)
ROC_knn <- roc(ds.validation.modelsel$DEP_DEL, validation_pred_prob_knn[,1])
ROC_pls <- roc(ds.validation.modelsel$DEP_DEL, validation_pred_prob_pls[,1])
ROC_lasso <- roc(ds.validation$DEP_DEL, validation_pred_prob_lasso[,1])
ROC_tree <- roc(ds.validation$DEP_DEL, validation_pred_prob_tree[,1])
ROC_glm <- roc(ds.validation$DEP_DEL, validation_pred_prob_glm[,1])
ROC_nb <- roc(ds.validation$DEP_DEL, nb_posterior_validation[,1])
ROC_nnet <- roc(ds.validation$DEP_DEL, validation_pred_prob_nnet[,1])
plot(ROC_knn, type = "S", col="orange")
plot(ROC_pls, type = "S", col="red", add=TRUE)
plot(ROC_lasso, type = "S", col="yellow", add=TRUE)
plot(ROC_tree, type = "S", col="green", add=TRUE)
plot(ROC_glm, type = "S", col="violet", add=TRUE)
plot(ROC_nb, type = "S", col="blue", add=TRUE)
plot(ROC_nnet, type = "S", col="gray", add=TRUE) 
#AUC
ROC_knn; ROC_pls; ROC_lasso; ROC_tree; ROC_glm; ROC_nb; ROC_nnet

# Lift curves to compare Naive Bayes and GLM
library(funModeling)
ds.validation$nb_posterior_validation <- nb_posterior_validation[,1]
ds.validation$validation_pred_prob_glm <- validation_pred_prob_glm[,1]
validation_elements <- ds.validation[,c(23,24,25)]
gain_lift(data = validation_elements, score = 'nb_posterior_validation', target = 'DEP_DEL')
gain_lift(data = validation_elements, score = 'validation_pred_prob_glm', target = 'DEP_DEL')
# Naive Bayes is the winning model!

#STEP 3: Tune the classification threshold using the metric of interest (sensitivity)
library(ROCR)
predRoc <- prediction(nb_posterior_validation[,1], ds.validation$DEP_DEL)
sens.perf = performance(predRoc, measure = "sens")
plot(sens.perf)
#Display sensitivity in threshold increments of 0.01
cutoffs <- sens.perf@x.values[[1]]
sensitivity <- sens.perf@y.values[[1]]
cutoff_seq <- seq(0, 1, by = 0.01)
sens_interpolated <- approx(cutoffs, sensitivity, xout = cutoff_seq, rule = 2)$y
sens_df <- data.frame(Cutoff = cutoff_seq, Sensitivity = sens_interpolated)
head(sens_df)

#Evaluating the confusion matrix with 2 possible thresholds
pred_y_nb_1 <- ifelse(nb_posterior_validation[,1]>0.01, "Delay","On_time")
pred_y_nb_1 <- as.factor(pred_y_nb_1)
confusionMatrix(pred_y_nb_1, ds.validation$DEP_DEL)
pred_y_nb_2 <- ifelse(nb_posterior_validation[,1]>0.02, "Delay","On_time")
pred_y_nb_2 <- as.factor(pred_y_nb_2)
confusionMatrix(pred_y_nb_2, ds.validation$DEP_DEL)

#STEP 4: Predict the target for a new dataset
score_prob = predict(naive_all, score_data, type="prob")
head(score_prob)
prob_delay=score_prob$posterior
score_data$pred_y=ifelse(prob_delay[,1]>0.01, "Delay","On_time")
head(score_data)
score_data$pred_y <- as.factor(score_data$pred_y)
confusionMatrix(score_data$pred_y, score_data$DEP_DEL) # Only 7 misclassifications
