# Artificial Neural Networks using Keras
# Simple Classification
# by Joe Davis

#' -------------------------------------------------------------------------- #
# 1. Load the Required libraries  
#' -------------------------------------------------------------------------- #
#' 
#' #' #install.packages('package_name')
library(tidyverse) # core library
library(tidyr) # for piping %>%
library(dplyr) # for data wrangling such as mutate, subset, filter, etc 
library(stringr) # for manipulating strings
library(ggplot2) # for making visualization/plot

# ML/Support libraries
library(keras) # tensorflow wrapper / API
library(tensorflow) # ML engine
#library(tfdatasets) # Tensorflow provided data sets are in here
library(reticulate) # embeds a Python session into R for using Python packages
library(caret) # for Classification and Regression training 
#library(tidymodels) # for model creation and machine learning
#library(text2vec) # NLP ML


# Other libraries
#library(coro) # functions that can be suspended/resumed later
#library(recipes) # transforming data (strings to numeric), modeling building
#library(GGally) # this is for inspecting data & making tables, ext of ggplot2
#library(skimr) # this is for reviewing overall statistics
#library(purrr) # for applying functions
#library(readxl) # reads in Microsoft Excel files
#library(rmarkdown) # producing end result reports (pmydata, html, export)

# Text Mining and String/Text Manipulation
#library(tm) # Text Mining library
#library(textstem) # for lemmatisation 
#library(tokenizers) # for tokenizing words
#ibrary(textcat) # language identifier
#library(textstem) # text formatting
#library(tidytext) # text formatting to and from tidy formats
#ibrary(textmineR) # creates the DTM

# Text function libraries
#library(wordcloud) # for making B&W wordClouds
#library(RColorBrewer) # dependency for WordClouds
#library(wordcloud2) # for making COLOR wordClouds
#library(htmlwidgets) # dependency for WordClouds

# Create a temp file if needed
#logdir <- tempfile()

# Set the working directory
#setwd('C:/Users/....') 

# Grab a file from the working directory
#fileName <- read.csv("xxxxx.csv")

# Grabbing CSV files easy
#fileName <- file.choose()
#data <- read.csv("________.csv")

#' -------------------------------------------------------------------------- #
# 2. Load the Data
#' -------------------------------------------------------------------------- #

# Total number of samples is 2000
# Create three separate groups (a,b,c)
n=2000
a <- sample(1:20, n, replace = T)
b <- sample(1:50, n, replace = T)
c <- sample(1:100, n, replace = T)

# Determine color assignment (which will be predicted later) based on values
# Create a dataframe (mydata) that will hold the data
flag <- ifelse(a > 15 & b > 30 & c > 60, "red", 
               ifelse(a<=9 & b<25& c<=35, "yellow", "green"))
mydata <- data.frame(a = a,
                 b = b, 
                 c = c, 
                 flag = as.factor(flag))

#' -------------------------------------------------------------------------- #
# 3. Pre-Processing Data
#' -------------------------------------------------------------------------- #

# Split data into train and test partitions (90 - 10 split).
indexes = sample(1:nrow(mydata), size = .90 * nrow(mydata))
train <- mydata[indexes, ]
test <- mydata[-indexes, ]

# Create the Training & Testing data with the above partitions
train.x <- as.matrix(train[ , 1:3])
train.y <- to_categorical(matrix(as.numeric(train[ ,4])-1))
test.x <- as.matrix(test[ , 1:3])
test.y <- to_categorical(matrix(as.numeric(test[ ,4])-1))

#' -------------------------------------------------------------------------- #
# 4. Create the model
#' -------------------------------------------------------------------------- #

# Create Keras model
model <- keras_model_sequential() %>%
  # Input layer with shape (number of X variables)
  layer_dense(units=64, activation = "relu", input_shape = c(3)) %>% 
  # Output layer with the units reflecting the total number of possible outputs
  layer_dense(units =3, activation = "softmax")

# View the model
summary(model)

# Compile the model
model %>% compile(optimizer = "rmsprop", 
                  loss = "categorical_crossentropy",
                  metric = c("accuracy"))

# Fit the model
model %>% keras::fit(
  train.x, train.y, 
  epochs = 100, 
  batch_size = 25)

# Make a prediction
pred <- model %>% predict(test.x) 

#' -------------------------------------------------------------------------- #
# 5. Evaluate the model
#' -------------------------------------------------------------------------- #

# Accuracy
scores <- model %>% evaluate(test.x, test.y)

# Formatting output decimals
pred <- format(round(pred, 2), nsamll = 4)

# Summary data frame for ease of display
result <- data.frame("green" = pred[,1], 
                     "red" = pred[,2], 
                     "yellow" = pred[,3], 
                     "predicted" = ifelse(max.col(pred[ ,1:3]) == 1, "green",
                                          ifelse(max.col(pred[ ,1:3]) == "2", "red", "yellow")),
                     original = test[ ,4])

# Create a caret styled Confusion Matrix w/additional statistics
table <- table(result$predicted, result$original)
cfm <- confusionMatrix(table)

# Output all the results
print(head(result))
print(scores)
print(cfm)
