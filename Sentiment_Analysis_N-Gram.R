# Sentiment & N-Gram Analysis
# by Joe Davis

#' -------------------------------------------------------------------------- #
# 1. Load the Required libraries  
#' -------------------------------------------------------------------------- #

#install.packages('package_name')
library(tidyverse) # core library
library(tidyr) # for piping %>%
library(dplyr) # for data wrangling such as mutate, subset, filter, etc 
library(stringr) # for manipulating strings
library(ggplot2) # for making visualization/plot

# ML/Support libraries
#library(keras) # tensorflow wrapper / API
#library(tensorflow) # ML engine
#library(tfdatasets) # Tensorflow provided data sets are in here
#library(reticulate) # embeds a Python session into R for using Python packages
#library(caret) # for Classification and Regression training 
#library(tidymodels) # for model creation and machine learning
#library(text2vec) # NLP ML

# Other libraries
#library(coro) # functions that can be suspended/resumed later
#library(recipes) # transforming data (strings to numeric), modeling building
#library(GGally) # this is for inspecting data & making tables, ext of ggplot2
#library(skimr) # this is for reviewing overall statistics
#library(purrr) # for applying functions
library(readxl) # reads in Microsoft Excel files
#library(rmarkdown) # producing end result reports (pmydata, html, export)

# Text Mining and String/Text Manipulation
#library(tm) # Text Mining library
library(textstem) # for lemmatisation, text formatting
#library(tokenizers) # for tokenizing words
library(textcat) # language identifier
library(tidytext) # text formatting to tidy formats
#ibrary(textmineR) # creates the DTM
library(textdata)

# Text function libraries
#library(wordcloud) # for making B&W wordClouds
#library(RColorBrewer) # dependency for WordClouds
#library(wordcloud2) # for making COLOR wordClouds
#library(htmlwidgets) # dependency for WordClouds

#Set the working directory and Load in the data set
# LAPTOP working directory
#setwd('C:/Users/Joe Davis/Desktop/R Projects/HW5')
# DESKTOP working directory
setwd('C:/Users/Joe Davis/Desktop/R Projects')

# Grab a file from the working directory
#fileName <- read.csv("xxxxx.csv")

# Grabbing CSV files easy
#fileName <- file.choose()

#' -------------------------------------------------------------------------- #
# 2. Load the Data
#' -------------------------------------------------------------------------- #

# STEP 1: Import the data, given in MS Excel format, into R verify the number
# of sheets within the MS Excel spreadsheet-call out correct sheet.
# This helper function, loading the MS Excel file, is too long and needs to be
# shortened to fit within the PEP8 guidelines.
imdb_data <- read_excel("C:\\Users\\Joe Davis\\Documents\\MHCC UoP\\U of P\\Summer 2022\\BUS591F Text Analytics\\HW\\HW2\\IMDB_Dataset.xlsx")

# Setup a working file, in case mistakes are made and work is overwritten.
mydata <- imdb_data
mydata <- as.data.frame(mydata)

#' -------------------------------------------------------------------------- #
# 3. Pre-Processing Data
#' -------------------------------------------------------------------------- #

# Remove/replace the numerics, symbols, punctuation and new lines.
# First, remove new lines using a few lines of code that will be applicable
# depending on the source of the documnet (Mac OS, Windows, Linux, etc).
mydata$review <- str_replace_all(mydata$review, "[\\r\\n]", " ")

# Next, replace all the numeric values.
mydata$review <- str_replace_all(mydata$review, "[\\r\\d]", " ")

# Next, replace all the punctuation.
mydata$review <- str_replace_all(mydata$review, "[:punct:]", " ")

# Next, replace all the symbols.
mydata$review <- str_replace_all(mydata$review, "[\\r\\W]", " ")

# Next, make all the characters lower case.
mydata$review <- str_to_lower(mydata$review)

# Next, remove extraneous white space (extra spaces).
mydata$review <- str_squish(mydata$review)

# Create a new column for 'language', like an index.  
# This step can be slow, be prepared to step away for a bit.
# Remove all reviews that are not in English (long processing time).
mydata$language <- textcat(mydata$review)

# Create new working df for backup purposes - delete later if need be.
mydata1 <- mydata[mydata$language == "english", c("review", "sentiment")]
mydata1 <- as.data.frame(mydata1)

# "afinn' method - this lexicon has values from -5:5 already assigned
#' -------------------------------------------------------------------------- #
# 4. Tokenize the Data into individual strings
#' -------------------------------------------------------------------------- #

# Get the lexicon to be used for sentiment analysis - 'afinn'
# Other options include:  'nrc' and 'bing'
sentiment_dictionary1 <- get_sentiments("afinn")

# Create a new variable for the individual tokens, create a new column
# called "reviewId" so we can group the tokens by the source/single review,
# then break out the individual words, remove the stop_words and use inner_join
# with the sentiment_dictionary to apply a value from 5 to -5 to each token.
mydata1_tokens <- mydata1 %>% 
  mutate(reviewId= row_number()) %>%
  unnest_tokens(word, review) %>%
  anti_join(stop_words, by = "word") %>%
  inner_join(sentiment_dictionary1, by = "word")

# Sum up all the tokens by 'reviewId' to get a numeric sentiment value.
# Compare against the given "sentiment" value from the sheet.
mydata1_sentiment <- mydata1_tokens %>%
  group_by(reviewId) %>%
  summarize(value = sum(value))

# Join two data frame aspects together (review Id and the sentiment total)
mydata_final1 <- mydata1 %>%
  mutate(reviewId = row_number()) %>%
  inner_join (mydata1_sentiment, by = "reviewId")

#' -------------------------------------------------------------------------- #
# 5. Filter positive and negative sentiments & compute accuracy
#' -------------------------------------------------------------------------- #

# Mutate the value column into a string of either "positive" or "negative"
# based on the numeric value (pos > 0, neg <= 0) then compare the value and
# sentiment columns for accuracy (1 = matching, 0 = not matching)
mydata_final1$valuesent <- if_else(mydata_final1$value > 0, 'positive', 
                                   'negative')
mydata_final1$accuracy <- if_else(mydata_final1$valuesent == 
                                    mydata_final1$sentiment, 1, 0)

# Compute the accuracy %.
acc1 <- (sum(mydata_final1$accuracy) / nrow(mydata_final1)) * 100
accuracy_final1 <- c('Accuracy is ', acc1, '%')
accuracy_final1

#' -------------------------------------------------------------------------- #
# 6. Filter positive and negative sentiments and plot
#' -------------------------------------------------------------------------- #

# Count the number of correct predictions vs incorrect per sentiment choice.
# Will be using these values to create a data frame that will be plotted.
mydata1_np <- subset(mydata_final1, sentiment=="negative")
nn1 <- sum(mydata1_np$accuracy)
np1 <- nrow(mydata1_np) - nn1
mydata1_pp <- subset(mydata_final1, sentiment=="positive")
pp1 <- sum(mydata1_pp$accuracy)
pn1 <- nrow(mydata1_pp)- pp1

# Create a data frame that will hold the percent values for Confusion Matrix
cmat1 <- data.frame(true_value = c('negative','negative','positive', 'positive'),
                    predicted_value = c('negative', 'positive', 'negative', 
                                        'positive'),
                    Frequency = c(nn1, np1, pn1, pp1))

# Generate the Confusion matrix and plot w/ggplot2.
plotCM1 <- cmat1 %>% 
  inner_join( (cmat1 %>% 
                 group_by(true_value) %>%
                 summarize(total = sum(Frequency))),
              by = "true_value") %>%
  mutate(ratio = Frequency/total, 
         success_failure = ifelse(true_value == predicted_value, 
                                  "success", "failure")) %>%
  select(true_value, predicted_value, success_failure, ratio)

# Create the visual COnfusion Matrix plot
ggplot(plotCM1, 
       mapping = aes(x = true_value, y = predicted_value, 
                     fill = success_failure, 
                     alpha = ratio)) + 
  geom_tile() +
  geom_text(aes(label = paste0(round(ratio,2), "%"), vjust = 1, fontface  = 
                  "bold", alpha = 1)) +
  scale_fill_manual(values = c(success = "green", failure = "red")) +
  labs(title="Confusion Matrix - Frequency and ratio ") +
  theme_bw() + 
  theme(legend.position = "none")

# ------------------------Repeat with different lexicon -----------------------
# "nrc" method - will have to add the numeric values for neg/pos values
#' -------------------------------------------------------------------------- #
# 4. Tokenize the Data into individual strings
#' -------------------------------------------------------------------------- #

# Get the lexicon to be used for sentiment analysis - 'nrc'
# Other options include:  'afinn' and 'bing'
sentiment_dictionary2 <- get_sentiments("nrc")

# Create a new variable for the individual tokens, create a new column
# called "reviewId" so we can group the tokens by the source/single review,
# then break out the individual words, remove the stop_words and use inner_join.
mydata2_tokens <- mydata %>% 
  mutate(reviewId= row_number()) %>%
  unnest_tokens(word, review) %>%
  anti_join(stop_words, by = "word") %>%
  inner_join(sentiment_dictionary2, by = "word") 

# Removing some content ('sentiment.x' and language), reassigning column names
mydata2_tokens <- subset(mydata2_tokens, select = - sentiment.x)
mydata2_tokens <- mydata2_tokens[ ,2:4]
colnames(mydata2_tokens) <- c('reviewId', 'word', 'sentiment')

# Remove all sentiments keywords that are NOT "positive" or "negative".
# Two separate dfs were created and then combined into one
mydata2_pos_tokens <- subset(mydata2_tokens, sentiment == "positive")
mydata2_neg_tokens <- subset(mydata2_tokens, sentiment == "negative")
mydata2_tokens <- rbind(mydata2_pos_tokens, mydata2_neg_tokens)

#' -------------------------------------------------------------------------- #
# 5. Create Sentiment data frame with both positive and negative tokens
#' -------------------------------------------------------------------------- #

# Create sentiment data frame and sort by 'reviewId'.
mydata2_sentiment <- mydata2_tokens %>%
  group_by(reviewId)

# Convert to an integer values.
# Assign a +1 for positive values and a -1 for negative values.
mydata2_sentiment$valuesent <- if_else(mydata2_sentiment$sentiment == 
                                         "positive", 1, -1)

# Group and sum the totals of the tokens (words) by 'reviewId'.
mydata2_sentiment <- mydata2_sentiment %>%
  group_by(reviewId) %>%
  summarize(valuesent = sum(valuesent))

# Assign a subjective string value to a numeric > than 0.
mydata2_sentiment$predicted_value <- if_else(mydata2_sentiment$valuesent > 0, 
                                             "positive", "negative")

# Join the original data frame and mydata2_sentiment to compare the accuracy.
mydata_final2 <- mydata1 %>%
  mutate(reviewId = row_number()) %>%
  inner_join (mydata2_sentiment, by = "reviewId")

# Change the names of the columns for continuity in the 2 final data frames.
colnames(mydata_final1) <- c('reivew', 'sentiment', 'reviewId', 'value', 
                             'predicted sentiment', 'accuracy')
colnames(mydata_final2) <- c('reivew', 'sentiment', 'reviewId', 'value', 
                             'predicted sentiment')

#' -------------------------------------------------------------------------- #
# 6. Accuracy and Plot
#' -------------------------------------------------------------------------- #

# Compute accuracy for mydata_final2
mydata_final2$accuracy <- if_else(mydata_final2$`predicted sentiment` == 
                                    mydata_final2$sentiment, 1, 0)

# Count the number of correct predictions vs incorrect per sentiment choice.
# Will be using these values to create a data frame that will be plotted.
mydata2_np <- subset(mydata_final2, sentiment=="negative")
nn2 <- sum(mydata2_np$accuracy)
np2 <- nrow(mydata2_np) - nn2
mydata2_pp <- subset(mydata_final2, sentiment=="positive")
pp2 <- sum(mydata2_pp$accuracy)
pn2 <- nrow(mydata2_pp)- pp2

# Create a data frame that will hold the percent values for Confusion Matrix.
cmat2 <- data.frame(true_value = c('negative','negative','positive', 'positive'),
                    predicted_value = c('negative', 'positive', 'negative', 'positive'),
                    Frequency = c(nn2, np2, pn2, pp2))

# Generate the Confusion matrix and plot w/ggplot2.
# Plot the Confusion matrix for visual build below.
plotCM2 <- cmat2 %>% 
  inner_join( (cmat2 %>% 
                 group_by(true_value) %>%
                 summarize(total = sum(Frequency))),
              by = "true_value") %>%
  mutate(ratio = Frequency/total, 
         success_failure = ifelse(true_value == predicted_value, 
                                  "success", "failure")) %>%
  select(true_value, predicted_value, success_failure, ratio)

# Create the visual COnfusion Matrix plot.
ggplot(plotCM2, 
       mapping = aes(x = true_value, y = predicted_value, 
                     fill = success_failure, 
                     alpha = ratio)) + 
  geom_tile() +
  geom_text(aes(label = paste0(round(ratio,2), "%"), vjust = 1, fontface  = "bold", alpha = 1)) +
  scale_fill_manual(values = c(success = "green", failure = "red")) +
  labs(title="Confusion Matrix - Frequency and ratio ") +
  theme_bw() + 
  theme(legend.position = "none")

# Mutate the value column into a string of either "positive" or "negative"
# based on the numeric value (pos > 0, neg <= 0) then compare the value and
# sentiment columns for accuracy (1 = matching, 0 = not matching).
mydata_final2$valuesent <- if_else(mydata_final2$value > 0, 'positive', 'negative')
mydata_final2$accuracy <- if_else(mydata_final2$valuesent == mydata_final2$sentiment, 1, 0)

# Compute the accuracy %.
acc2 <- (sum(mydata_final2$accuracy) / nrow(mydata_final2)) * 100
accuracy_final2 <- c('Accuracy is ', acc2, '%')
accuracy_final2

# -----------------------------END OF CODE ------------------------------------
