#' -------------------------------------------------------------------------- '#
#'  Purpose: Create a Keras model in this R script to train and test the 
#'  classification model to predict spam and ham text messages. 
#'  Required packages - "keras" and "reticulate" for running neural networks. 
#'  "tm", "purrr", "stringr", "textstem" for text pre-processing and requires 
#'  other supporting libraries such as "dplyr" and "tidyr".
#'  Author: Joe Davis
#'  Date: 8/3/2022
#' -------------------------------------------------------------------------- '#
# Artificial Neural Networks using Keras
# Binary Classification
# by Joe Davis


#' -------------------------------------------------------------------------- #
# 1. Load the Required libraries  
#' -------------------------------------------------------------------------- #
#' 
#' #' #install.packages('package_name')
library(tidyverse) # standard add-ons
library(tidyr) # for piping %>%
library(dplyr) # mutate, subset, filter functions using %>% operations 
library(stringr) # for manipulating strings
library(ggplot2) # for making visualization/plot

# ML/Support libraries
library(keras) # tensorflow wrapper / API
library(tensorflow) # ML engine
library(tfdatasets) # tensorflow provided data sets are in here
library(reticulate) # for loading Python packages into R

# Other libraries
library(coro) # for correlation
library(tidymodels)
library(recipes) # transforming data (strings to numeric)
library(GGally) # this is for inspecting data & making tables
library(skimr) # this is for reviewing overall statistics
library(purrr) # for applying functions
library(readxl) # reads in Microsoft Excel files
library(rmarkdown) # producing end result reports

# Text Mining and String/Text Manipulation
library(tm) # Text Mining
library(textstem) # For lemmatisation 
library(tokenizers) # for tokenizing words
library(textcat) # Language identifier
library(textstem) # does something
library(tidytext) # does something
library(text2vec) # does stuff
library(textmineR) # creates the DTM

# Text function libraries
library(wordcloud) # for making B&W word clouds
library(RColorBrewer) # dependency for WordClouds
library(wordcloud2) # for making COLOR word clouds
library(htmlwidgets) # depenedency for WordClouds

# Tempfile
logdir <- tempfile()

#' -------------------------------------------------------------------------- #
# 2. Load the Data
#' -------------------------------------------------------------------------- #

#Set the working directory and Load in the data set

# LAPTOP working directory
#setwd('C:/Users/Joe Davis/Desktop/R Projects/HW5')
# DESKTOP working directory
setwd('C:/Users/Joe Davis/Desktop/R Projects') 

#fileName <- file.choose()
#data <- read.csv("________.csv")

# Get the data
data <- read.csv("Spam_Ham.csv")
spamDF <- data

#' -------------------------------------------------------------------------- #
# 3. Pre-Processing Data
#' -------------------------------------------------------------------------- #
#' The text field contains 'Subject: ' in the first 9 characters in all the 
#' emails. These characters are not required as they they make a low priority 
#' value for the token if we take the TF-IDF 
#' Minimum pre-processing steps include: 
#' i. converting to lower case
#' ii. Remove stopwords 
#' iii. Remove punctuations 
#' iv. Remove numbers 
#' v. Remove white spaces 
#' vi. Remove lemmas (plural words and keep the root words)
#' -------------------------------------------------------------------------- #

# remove the first 9 characters from the 'text' column
spamDF$text <- substring(spamDF$text, 9)

# Remove HTML tags from the string 
spamDF$text <- gsub("<.*?>", "", spamDF$text)

# Replace any character that is not alphanumeric with a space 
spamDF$text <- gsub("[^[:alnum:]=\\.]", " ", spamDF$text)

# Remove punctuation characters 
spamDF$text <- str_replace_all(spamDF$text, "[:punct:]","")

# Pre-process text using "tm" package 
text_corpus <- (VectorSource(spamDF$text))

# We work with corpus while using "tm" package
text_corpus <- Corpus(text_corpus)

# Text pre-processing using "tm" package
text_corpus <-tm_map(text_corpus , tolower)
text_corpus <- tm_map(text_corpus, removeNumbers)
text_corpus <- tm_map(text_corpus, stripWhitespace)
text_corpus <- tm_map(text_corpus, removeWords, stopwords('english'))

# Primarily the text data is saved inside content member of corpus object
spamDF$text <- text_corpus$content
spamDF$text <- str_squish(spamDF$text)
rm(text_corpus)

#' -------------------------------------------------------------------------- #
#' 4. Tokenize the text and take the top 10K words using text_tokenizer 
#' and fit_text_tokenizer to generate indexes 
#' -------------------------------------------------------------------------- #

tokenizer <- text_tokenizer(num_words = 10000) %>%
  fit_text_tokenizer(spamDF$text)
word_index <- tokenizer$word_index
max_words <- 10000

#' -------------------------------------------------------------------------- #
#' 5. Each text email has to be converted to the corresponding word indexes 
#' Use texts_to_sequences function to generate the texts as integer indexes  
#' Better you create the indexes as a separate column in the spamDF dataframe 
#' -------------------------------------------------------------------------- #

spamDF$text_index <- texts_to_sequences(tokenizer, spamDF$text)

#' -------------------------------------------------------------------------- #
#' 6. Split the dataframe into training and test using a 70% - 30% split 
#' -------------------------------------------------------------------------- #
#'  
row_selection <- sample(c(0,1), nrow(spamDF), replace = TRUE, prob = c(0.7, 0.3))
spamDF_train <- spamDF[row_selection == 0,]
spamDF_test <- spamDF[row_selection == 1,]

#' -------------------------------------------------------------------------- #
#' 7. Create x_train, x_test, y_train, y_test
#' Make sure that they are not dataframes. Keras libraries work great with
#' integer matrices and vectors.
#' Also, there should not be any colnames for the matrices 
#' 
#' Hint:  
#' x_train <- spamDF_train[5] 2nd column of the dataframe contains integer 
#' indices of the words used in the email messages
#' colnames(x_train) <- NULL   
#' x_train <- sapply(x_train, as.list) # Convert the dataframe column as list of
#' integer lists. The x_train should contain multiple list of integers e.g, if 
#' the x_train contains 1500 text messages, then x_train should be containing 
#' 1500 rows of varying length integer vectors. You can compare the indexes with 
#' the pre-loaded indices of dataset_imdb 
#' 
#' y_train should be a simple integer vector 
#' y_train <- spamDF_train[1]
#' y_train <- sapply(y_train[1], as.integer) # Converts 
#' y_train <- c(y_train) # Converts a dataframe column to vector 
#' colnames(y_train) <- NULL 
#' -------------------------------------------------------------------------- #
#'  

# Create the training & testing variables
# X train - another option for assigning data without a loop 
x_Train <- sapply(spamDF_train[5], as.list)
x_Train <- lapply(x_Train, as.integer)
x_Train1 <- pad_sequences(x_Train, maxlen = sentlen)
x_Train1[2,]
x_Train[[2]]

# X Training Data - using a for loop to get a list of lists as a result
x_train <- list()
df <- spamDF_train[5]
colnames(df) <- NULL
for(i in 1:ncol(df)) {
  x_train[i] <- list(df[ ,i])
}
x_train <- x_train[[1]]

# X Testing data
x_test <- list()
df1 <- spamDF_test[5]
for(i in 1:ncol(df1)) {
  x_test[i] <- list(df1[ ,i])
}
x_test <- x_test[[1]]

# Y Training Data
y_train <- list()
df2 <- spamDF_train[4]
colnames(df2) <- NULL
for(i in 1:ncol(df2)) {
  y_train[i] <- list(df2[ ,i])
}
y_train <- y_train[[1]]

# Y Testing data
y_test <- list()
df3 <- spamDF_test[4]
for(i in 1:ncol(df3)) {
  y_test[i] <- list(df3[ ,i])
}
y_test <- y_test[[1]]

#' -------------------------------------------------------------------------- #
#' 8. Run pad_sequences with the mean number of sentences length. 
#' For example, if the indexes are saved in the dataframe column text_indexes 
#' then the summary of the sentences length can be found using the summary as: 
#' summary(sapply(spamDF$text_indexes, length)) 
#' -------------------------------------------------------------------------- #
#'  
# look at the length of the reviews & set max length to the AVG length +-
summary(sapply(spamDF$text_index, length)) 
sentlen <- 100

# take the top AVG# of words from each review
x_train <- pad_sequences(x_train, maxlen = sentlen)
x_test <- pad_sequences(x_test, maxlen = sentlen)

#' -------------------------------------------------------------------------- #
#' 9. Create a simple model with layer_embedding, layer_flatten and classification 
#' Compile the model 
#' Generate the model fit 
#' Evaluate the model 
#' Make predictions
#' Report the accuracy 
#' -------------------------------------------------------------------------- #
#'  

# -------------------------- Word2Vec w/flatten --------------------------

# Create Keras model
model <- keras_model_sequential() %>%
  # Word Embedding
  layer_embedding(input_dim = 10000, output_dim = 32, input_length = sentlen) %>%
  # We can pass the layer embedding directly as input to RNN nodes
  #layer_simple_rnn(units = 32, activation = 'tanh') %>%
  layer_flatten() %>%
  layer_dense(units = 1, activation = 'sigmoid')

# checkout the the simple_rnn inputs
#?layer_simple_rnn()
summary(model)

# Compile the model
model %>% compile(
  optimizer = 'adam',
  loss = 'binary_crossentropy', 
  metrics = c('accuracy')
)

# Fit the model
model %>% keras::fit(
  x_train, y_train,
  epochs = 10,
  batch_size = 32,
  validation_split = 0.2
)

# Evaluate the model
model %>% keras::evaluate(
  x = x_test,
  y = y_test,
  batch_size = 32,
  verbose = getOption('keras.fit_verbose', default = 1),
  sample_weight = NULL,
  steps = NULL,
  callbacks = NULL
)

# Prediction
Y_Predict <- model %>%
  predict(x_test)
Y_Predict <- ifelse(Y_Predict > 0.5, 1, 0)

# Confusion Matrix
confusionMatrix <- table(Y_Predict, y_test)
confusionMatrix

# Accuracy
acc <- (confusionMatrix[1,1] + confusionMatrix[2,2]) / sum(confusionMatrix)
acc

#' -------------------------------------------------------------------------- #
#' 10. Repeat the steps and create models using layer_simple_rnn and capture 
#' accuracy. Then layer_lstm, compile, fit, predict, and capture accuracy.
#' Finally, create a bidirectional lstm model, compile, fit, predict, and 
#' capture accuracy. 
#' -------------------------------------------------------------------------- #
#'  

# -------------------------------- RNN Model ----------------------------------
#                         Recurrent Neural Network

# Create Keras - RNN model (time-series)
model1 <- keras_model_sequential() %>%
  # Word Embedding
  layer_embedding(input_dim = 10000, output_dim = 32, input_length = sentlen) %>%
  # We can pass the layer embedding directly as input to RNN nodes
  # try to match the units w/ the output_dim if possible
  
  # We will not have layer_embedding in case of numeric predictors to predict demand
  # We do not have output_dim (Hidden embedding matrix)
  # e.g., our sample size is 1000 observations for the 30 predictors
  # RNN expects you to pass different inputs to different hidden units. 
  # 10 RNN hidden units 
  # Time series should know how many instances of input do you have 
  # 1000 * 30 * 10 
  
layer_simple_rnn(units = 16, activation = 'tanh') %>%
  layer_dense(units = 1, activation = 'sigmoid')

?layer_simple_rnn
# checkout the the simple_rnn inputs
#?layer_simple_rnn()
summary(model1)

# Compile the model
model1 %>% compile(
  optimizer = 'adam',
  loss = 'binary_crossentropy', 
  metrics = c('accuracy')
)

# Fit the model
model1 %>% keras::fit(
  x_train, y_train,
  epochs = 10,
  batch_size = 32,
  validation_split = 0.2
)

# Evaluate the model
model1 %>% keras::evaluate(
  x = x_test,
  y = y_test,
  batch_size = 32,
  verbose = getOption('keras.fit_verbose', default = 1),
  sample_weight = NULL,
  steps = NULL,
  callbacks = NULL
)

# Prediction
Y_Predict1 <- model1 %>%
  predict(x_test)
Y_Predict1 <- ifelse(Y_Predict1 > 0.5, 1, 0)

# Confusion Matrix
confusionMatrix1 <- table(Y_Predict1, y_test)
confusionMatrix1

# Accuracy
acc1 <- (confusionMatrix1[1,1] + confusionMatrix1[2,2]) / sum(confusionMatrix1)
acc1

# ------------------------------- LSTM Model ---------------------------------
#                             Long Short Term Memory

# Create Keras - LSTM model
model2 <- keras_model_sequential() %>%
  # Word Embedding
  layer_embedding(input_dim = 10000, output_dim = 32, input_length = sentlen) %>%
  # We can pass the layer embedding directly as input to RNN nodes
  #layer_simple_rnn(units = 32, activation = 'tanh') %>%
  # LSTM layer for LSTM modeling
  layer_lstm(units = 64, activation = 'tanh') %>%
  #layer_flatten() %>%
  layer_dense(units = 1, activation = 'sigmoid')

# Inspect the model
summary(model2)

# Compile the model
model2 %>% compile(
  optimizer = 'adam',
  loss = 'binary_crossentropy', 
  metrics = c('accuracy')
)

# Fit the model
model2 %>% keras::fit(
  x_train, y_train,
  epochs = 10,
  batch_size = 32,
  validation_split = 0.2
)

# Evaluate the model
model2 %>% keras::evaluate(
  x = x_test,
  y = y_test,
  batch_size = 32,
  verbose = getOption('keras.fit_verbose', default = 1),
  sample_weight = NULL,
  steps = NULL,
  callbacks = NULL
)

# Prediction
Y_Predict2 <- model2 %>%
  predict(x_test)
Y_Predict2 <- ifelse(Y_Predict2 > 0.5, 1, 0)

# Confusion Matrix
confusionMatrix2 <- table(Y_Predict2, y_test)
confusionMatrix2

# Accuracy
acc2 <- (confusionMatrix2[1,1] + confusionMatrix2[2,2]) / sum(confusionMatrix2)
acc2


# ------------------------ Bidirectional LSTM Model ---------------------------
#                   Bidirectional Long Short Term Memory

# Create Keras - Bidirectional LSTM model
model3 <- keras_model_sequential() %>%
  # Word Embedding
  layer_embedding(input_dim = 10000, output_dim = 32, input_length = sentlen) %>%
  # We can pass the layer embedding directly as input to RNN nodes
  #layer_simple_rnn(units = 32, activation = 'tanh') %>%
  # LSTM layer for LSTM modeling
  #layer_lstm(units = 64, activation = 'tanh') %>%
  bidirectional(layer_lstm(units = 32)) %>%
  #layer_flatten() %>%
  layer_dense(units = 1, activation = 'sigmoid')

# Inspect the model
summary(model3)

# Compile the model
model3 %>% compile(
  optimizer = 'adam',
  loss = 'binary_crossentropy', 
  metrics = c('accuracy')
)

# Fit the model
model3 %>% keras::fit(
  x_train, y_train,
  epochs = 10,
  batch_size = 32,
  validation_split = 0.2
)

# Evaluate the model
model3 %>% keras::evaluate(
  x = x_test,
  y = y_test,
  batch_size = 32,
  verbose = getOption('keras.fit_verbose', default = 1),
  sample_weight = NULL,
  steps = NULL,
  callbacks = NULL
)

# Prediction
Y_Predict3 <- model3 %>%
  predict(x_test)
Y_Predict3 <- ifelse(Y_Predict3 > 0.5, 1, 0)

# Confusion Matrix
confusionMatrix3 <- table(Y_Predict3, y_test)
confusionMatrix3

# Accuracy
acc3 <- (confusionMatrix3[1,1] + confusionMatrix3[2,2]) / sum(confusionMatrix3)
acc3


