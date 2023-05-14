# 2nd Subtask Objectives...............

library(neuralnet)
library(dplyr)
library(readxl)
library(MLmetrics)
library(Metrics)


# b...............


# Load the data set
electricity_consumption <- read_excel("electricity_consumption.xlsx")
View(electricity_consumption)

# Replace the column names with new relevant column names
colnames(electricity_consumption) <- c("Date", "18", "19", "20") 
View(head(electricity_consumption))

# Create time delayed input variables
time_delayed <- bind_cols(
  T7 = lag(electricity_consumption$'20', 7),
  T4 = lag(electricity_consumption$'20', 4),
  T3 = lag(electricity_consumption$'20', 3),
  T2 = lag(electricity_consumption$'20', 2),
  T1 = lag(electricity_consumption$'20', 1),
  outputprediction = lag(electricity_consumption$'20', 0)
)

# Remove rows with missing values
time_delayed <- na.omit(time_delayed)

# Construction of different time-delayed input vectors and related i/o matrices.
T_1 <- cbind(time_delayed$T1, time_delayed$outputprediction)
colnames(T_1)<- c("Input", "Output")

T_2 <- cbind(time_delayed$T1, time_delayed$T2, time_delayed$outputprediction)
colnames(T_2)<- c("Input_1", "Input_2", "Output")

T_3 <- cbind(time_delayed$T1, time_delayed$T2, time_delayed$T3, time_delayed$outputprediction)
colnames(T_3)<- c("Input_1", "Input_2", "Input_3", "Output")

T_4 <- cbind(time_delayed$T1, time_delayed$T2, time_delayed$T3, time_delayed$T4, time_delayed$outputprediction)
colnames(T_4)<- c("Input_1", "Input_2", "Input_3", "Input_4", "Output")

T_5 <- cbind(time_delayed$T1, time_delayed$T2, time_delayed$T3, time_delayed$T4,time_delayed$T7, time_delayed$outputprediction)
colnames(T_5)<- c("Input_1", "Input_2", "Input_3", "Input_4", "Input_5", "Output")


# c...............

#Defining the normalize function (Min-Max Normalization)
norm <- function(x) {
  return ((x - min(x)) / (max(x)-min(x)))
}

#Normalizing the I/O Matrices
normT_1 <- norm(T_1)
normT_2 <- norm(T_2)
normT_3 <- norm(T_3)
normT_4 <- norm(T_4)
normT_5 <- norm(T_5)

# d...............

#define the training sets and testing sets for each I/O matrix
train_T_1 <- normT_1[1:380,]
test_T_1 <- normT_1[381: nrow(normT_1),]

train_T_2 <- normT_2[1:380,]
test_T_2 <- normT_2[381: nrow(normT_2),]

train_T_3 <- normT_3[1:380,]
test_T_3 <- normT_3[381: nrow(normT_3),]

train_T_4 <- normT_4[1:380,]
test_T_4 <- normT_4[381: nrow(normT_4),]

train_T_5 <- normT_5[1:380,]
test_T_5 <- normT_5[381: nrow(normT_5),]

# Training the Neural network for normT_1

# One hidden layer neural networks version #1

T_1_NN1 <- neuralnet(Output ~ Input, data = train_T_1, hidden = 4, linear.output = TRUE)
plot(T_1_NN1)

T_2_NN1 <- neuralnet(Output ~ Input_1 + Input_2, data = train_T_2, hidden = 4, linear.output = TRUE)
plot(T_2_NN1)

T_3_NN1 <- neuralnet(Output ~ Input_1 + Input_2 + Input_3, data = train_T_3, hidden = 4, linear.output = TRUE)
plot(T_3_NN1)

T_4_NN1 <- neuralnet(Output ~ Input_1 + Input_2 + Input_3 + Input_4, data = train_T_4, hidden = 4, linear.output = TRUE)
plot(T_4_NN1)

T_5_NN1 <- neuralnet(Output ~ Input_1 + Input_2 + Input_3 + Input_4 + Input_5, data = train_T_5, hidden = 4, linear.output = TRUE)
plot(T_5_NN1)


# Two hidden layer Neural networks

T_1_NN2 <- neuralnet(Output ~ Input, data = train_T_1, hidden = c(4,4), linear.output = TRUE)
plot(T_1_NN2)

T_2_NN2 <- neuralnet(Output ~ Input_1 + Input_2, data = train_T_2, hidden =c (4,4), linear.output=TRUE)
plot(T_2_NN2)

T_3_NN2 <- neuralnet(Output ~ Input_1 + Input_2 + Input_3, data = train_T_3, hidden = c(4,4), linear.output = TRUE)
plot(T_3_NN2)

T_4_NN2 <- neuralnet(Output ~ Input_1 + Input_2 + Input_3 + Input_4, data = train_T_4, hidden = c(4,4), linear.output = TRUE)
plot(T_4_NN2)

T_5_NN2 <- neuralnet(Output ~ Input_1 + Input_2 + Input_3 + Input_4 + Input_5, data = train_T_5, hidden = c(4,4), linear.output = TRUE)
plot(T_5_NN2)


# One hidden layer neural networks version #2

T_1_NN3 <- neuralnet(Output ~ Input, data = train_T_1, hidden = 7, linear.output = TRUE)
plot(T_1_NN3)

T_2_NN3 <- neuralnet(Output ~ Input_1 + Input_2, data = train_T_2, hidden = 7, linear.output = TRUE)
plot(T_2_NN3)

T_3_NN3 <- neuralnet(Output ~ Input_1 + Input_2 + Input_3, data = train_T_3, hidden = 7, linear.output = TRUE)
plot(T_3_NN3)

T_4_NN3 <- neuralnet(Output ~ Input_1 + Input_2 + Input_3 + Input_4, data = train_T_4, hidden = 7, linear.output = TRUE)
plot(T_4_NN3)

T_5_NN3 <- neuralnet(Output ~ Input_1 + Input_2 + Input_3 + Input_4 + Input_5, data = train_T_5, hidden = 7, linear.output = TRUE)
plot(T_5_NN3)


# Using performance indicators to determine the optimal NN topologies

# Calculation of the actual output of each I/O matrix's testing data

T_1_act_output <- test_T_1[, "Output"]
T_2_act_output <- test_T_2[, "Output"]
T_3_act_output <- test_T_3[, "Output"]
T_4_act_output <- test_T_4[, "Output"]
T_5_act_output <- test_T_5[, "Output"]

#Then the predicted output from each model is calculated

T_1_pred_output1 <- predict(object = T_1_NN1, test_T_1)
T_1_pred_output2 <- predict(object = T_1_NN2, test_T_1)
T_1_pred_output3 <- predict(object = T_1_NN3, test_T_1)

T_2_pred_output1 <- predict(object = T_2_NN1, test_T_2)
T_2_pred_output2 <- predict(object = T_2_NN2, test_T_2)
T_2_pred_output3 <- predict(object = T_2_NN3, test_T_2)

T_3_pred_output1 <- predict(object = T_3_NN1, test_T_3)
T_3_pred_output2 <- predict(object = T_3_NN2, test_T_3)
T_3_pred_output3 <- predict(object = T_3_NN3, test_T_3)

T_4_pred_output1 <- predict(object = T_4_NN1, test_T_4)
T_4_pred_output2 <- predict(object = T_4_NN2, test_T_4)
T_4_pred_output3 <- predict(object = T_4_NN3, test_T_4)

T_5_pred_output1 <- predict(object = T_5_NN1, test_T_5)
T_5_pred_output2 <- predict(object = T_5_NN2, test_T_5)
T_5_pred_output3 <- predict(object = T_5_NN3, test_T_5)

# Define the unnormalize function
unnormalize <- function(x, min_val, max_val) {
  return (x * (max_val - min_val) + min_val)
}

# Unnormalize the predicted outputs and actual outputs for each model

T_1_pred_output1_unnorm <- unnormalize(T_1_pred_output1, min(train_T_1[, "Output"]), max(train_T_1[, "Output"]))
T_1_pred_output2_unnorm <- unnormalize(T_1_pred_output2, min(train_T_1[, "Output"]), max(train_T_1[, "Output"]))
T_1_pred_output3_unnorm <- unnormalize(T_1_pred_output3, min(train_T_1[, "Output"]), max(train_T_1[, "Output"]))

T_1_act_output_unnorm <- unnormalize(T_1_act_output, min(train_T_1[, "Output"]), max(train_T_1[, "Output"]))

T_2_pred_output1_unnorm <- unnormalize(T_2_pred_output1, min(train_T_2[, "Output"]), max(train_T_2[, "Output"]))
T_2_pred_output2_unnorm <- unnormalize(T_2_pred_output2, min(train_T_2[, "Output"]), max(train_T_2[, "Output"]))
T_2_pred_output3_unnorm <- unnormalize(T_2_pred_output3, min(train_T_2[, "Output"]), max(train_T_2[, "Output"]))

T_2_act_output_unnorm <- unnormalize(T_2_act_output, min(train_T_2[, "Output"]), max(train_T_2[, "Output"]))

T_3_pred_output1_unnorm <- unnormalize(T_3_pred_output1, min(train_T_3[, "Output"]), max(train_T_3[, "Output"]))
T_3_pred_output2_unnorm <- unnormalize(T_3_pred_output2, min(train_T_3[, "Output"]), max(train_T_3[, "Output"]))
T_3_pred_output3_unnorm <- unnormalize(T_3_pred_output3, min(train_T_3[, "Output"]), max(train_T_3[, "Output"]))

T_3_act_output_unnorm <- unnormalize(T_3_act_output, min(train_T_3[, "Output"]), max(train_T_3[, "Output"]))

T_4_pred_output1_unnorm <- unnormalize(T_4_pred_output1, min(train_T_4[, "Output"]), max(train_T_4[, "Output"]))
T_4_pred_output2_unnorm <- unnormalize(T_4_pred_output2, min(train_T_4[, "Output"]), max(train_T_4[, "Output"]))
T_4_pred_output3_unnorm <- unnormalize(T_4_pred_output3, min(train_T_4[, "Output"]), max(train_T_4[, "Output"]))

T_4_act_output_unnorm <- unnormalize(T_4_act_output, min(train_T_4[, "Output"]), max(train_T_4[, "Output"]))

T_5_pred_output1_unnorm <- unnormalize(T_5_pred_output1, min(train_T_5[, "Output"]), max(train_T_5[, "Output"]))
T_5_pred_output2_unnorm <- unnormalize(T_5_pred_output2, min(train_T_5[, "Output"]), max(train_T_5[, "Output"]))
T_5_pred_output3_unnorm <- unnormalize(T_5_pred_output3, min(train_T_5[, "Output"]), max(train_T_5[, "Output"]))

T_5_act_output_unnorm <- unnormalize(T_5_act_output, min(train_T_5[, "Output"]), max(train_T_5[, "Output"]))

#Introduce a function to find performance metrics

perform_metrics <- function(act_output, pred_output){
  return(list(RMSE = rmse(act_output, pred_output),
              MAE = mae(act_output, pred_output),
              MAPE = mape(act_output, pred_output),
              SMAPE = smape(act_output, pred_output)))
}

#  calculion of performance metrics for each model

T_1_NN1_performance <- perform_metrics(T_1_act_output, T_1_pred_output1)
T_1_NN2_performance <- perform_metrics(T_1_act_output, T_1_pred_output2)
T_1_NN3_performance <- perform_metrics(T_1_act_output, T_1_pred_output3)

T_2_NN1_performance <- perform_metrics(T_2_act_output, T_2_pred_output1)
T_2_NN2_performance <- perform_metrics(T_2_act_output, T_2_pred_output2)
T_2_NN3_performance <- perform_metrics(T_2_act_output, T_2_pred_output3)

T_3_NN1_performance <- perform_metrics(T_3_act_output, T_3_pred_output1)
T_3_NN2_performance <- perform_metrics(T_3_act_output, T_3_pred_output2)
T_3_NN3_performance <- perform_metrics(T_3_act_output, T_3_pred_output3)

T_4_NN1_performance <- perform_metrics(T_4_act_output, T_4_pred_output1)
T_4_NN2_performance <- perform_metrics(T_4_act_output, T_4_pred_output2)
T_4_NN3_performance <- perform_metrics(T_4_act_output, T_4_pred_output3)

T_5_NN1_performance <- perform_metrics(T_5_act_output, T_5_pred_output1)
T_5_NN2_performance <- perform_metrics(T_5_act_output, T_5_pred_output2)
T_5_NN3_performance <- perform_metrics(T_5_act_output, T_5_pred_output3)

T_1_NN1_performance
T_1_NN2_performance
T_1_NN3_performance

T_2_NN1_performance
T_2_NN2_performance
T_2_NN3_performance

T_3_NN1_performance
T_3_NN2_performance
T_3_NN3_performance

T_4_NN1_performance
T_4_NN2_performance
T_4_NN3_performance

T_5_NN1_performance
T_5_NN2_performance
T_5_NN3_performance

# g...............

# Calculate total number of weight parameters for T_5_NN3
T_5_NN3_weights <- sum(sapply(T_5_NN3$weights, function(x) length(x)))
T_5_NN3_weights

# Calculate total number of weight parameters for T_5_NN2
T_5_NN2_weights <- sum(sapply(T_5_NN2$weights, function(x) length(x)))
T_5_NN2_weights
