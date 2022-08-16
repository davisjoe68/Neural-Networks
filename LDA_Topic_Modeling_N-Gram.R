# LDA Topic Modeling
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
library(tm) # Text Mining library
library(textstem) # for lemmatisation, text formatting
library(tokenizers) # for tokenizing words
library(textcat) # language identifier
library(tidytext) # text formatting to tidy formats
library(textmineR) # creates the DTM
library(textdata)

# Text function libraries
#library(wordcloud) # for making B&W wordClouds
#library(RColorBrewer) # dependency for WordClouds
#library(wordcloud2) # for making COLOR wordClouds
#library(htmlwidgets) # dependency for WordClouds

#' -------------------------------------------------------------------------- #
# 2. Load & Partition the Data
#' -------------------------------------------------------------------------- #

#Set the working directory and Load in the data set

# LAPTOP working directory
#setwd('C:/Users/Joe Davis/Desktop/R Projects/HW5')
# DESKTOP working directory
setwd('C:/Users/Joe Davis/Desktop/R Projects') 

#fileName <- file.choose()
#data <- read.csv("________.csv")

# Get the data
origdata <- read_csv("C:\\Users\\Joe Davis\\Desktop\\R Projects\\Misinformation_Tweets.csv")
mydata <- origdata

#' -------------------------------------------------------------------------- #
# 3. Pre-Processing Data
#' -------------------------------------------------------------------------- #

# Remove the columns we are not going to use.
mydata = mydata[ , c(3,8)]

# Rename the columns for ease of use
colnames(mydata) <- c('Tweet', 'Harm')

# Remove new lines from the strings (if any), numerics, punctuation ,etc
mydata$Tweet <- str_replace_all(mydata$Tweet, "[\\r\\n]", " ")
mydata$Tweet <- str_replace_all(mydata$Tweet, "[\\r\\d]", " ")
mydata$Tweet <- str_replace_all(mydata$Tweet, "[:punct:]", " ")
mydata$Harm <- str_replace_all(mydata$Harm, "[:punct:]", " ")
mydata$Tweet <- str_replace_all(mydata$Tweet, "[\\r\\W]", " ")
mydata$Tweet <- str_to_lower(mydata$Tweet)
mydata$Tweet <- str_squish(mydata$Tweet)

# tweetId
mydata$tweetId <- seq.int(nrow(mydata))

# Extract High Harm classified tweets
hiharm <- mydata %>%
  filter(Harm == "High Harm")

# Tokenize the words
hh_tokens <- hiharm %>%
  mutate(tweetId = row_number()) %>%
  unnest_tokens(word, Tweet) %>%
  anti_join(stop_words, by = "word") %>%
  count(word, tweetId)

# Lemmatize the words
hh_tokens$word <- lemmatize_words(hh_tokens$word, dictionary = lexicon::hash_lemmas)

# Assign the broken out tokens to the hh_DTM variable
hh_DTM <- hh_tokens %>%
  cast_dtm(tweetId, word, n)

#' -------------------------------------------------------------------------- #
# 4. Create LDA Topic Model
#' -------------------------------------------------------------------------- #

# Create the DTM (document term matrix) & remove stopwords
xdtm <- VCorpus(VectorSource(hiharm$Tweet)) %>%
  tm_map(removeWords, stopwords("en"))

# Create the DTM (document term matrix)
xdtm <- CreateDtm(
  doc_vec = hiharm$Tweet,
  doc_names = rownames(hiharm),
  ngram_window = c(1, 2), # Bi-grams instead of only words
  stopword_vec = c(stopwords::stopwords("en"), stopwords::stopwords(source = "smart")),
  lower = TRUE,
  remove_punctuation = TRUE,
  remove_numbers = TRUE,
  stem_lemma_function = NULL,
  verbose = FALSE)

# Fit the model
# Fit the LDA model model for the High-Harm tweets. Generate 15 topics and 
#' use 500 iterations while fitting the LDA topic model
#' Name the topic model you create as lda_misinformation_high
#  Apply the model to my xdtm data (15 topics and 500 iterations)
lda_misinformation_high <- FitLdaModel(
  xdtm,
  k = 15,
  iterations = 500,
  burnin = 200, # Hyper - parameters
  alpha = 0.1,
  beta = 0.05,
  optimize_alpha = TRUE,
  calc_likelihood = TRUE,
  calc_coherence = TRUE,
  calc_r2 = TRUE)

# Print the R^2 ("Arg squared"!)
lda_misinformation_high$r2

# Coherence histogram
# Topic Coherence (sum of the betas of all the words in each topic?)
sort(lda_misinformation_high$coherence)
lda_misinformation_high$coherence
hist(lda_misinformation_high$coherence)

# Capture the top 15 terms for each topic , Top M (qty) terms
lda_misinformation_high$top_Terms <- GetTopTerms(lda_misinformation_high$phi, 
                                                 M = 15, return_matrix = TRUE)
head(lda_misinformation_high$top_Terms)

#' ----------------------------------------------------------------------------
#' Assigning the labels for each topic using the set of topic keywords is the  
#' most important task of topic modeling. The textmineR package provides a simple 
#' implementation of assigning the topic lables using the best N-grams with the  
#' highest theta value. Phi is the word probability distribution per topic and 
#' theta is the topic-level probability distribution over the documents. 
#' Here in the below code, if the probability of a topic related for a particular 
#' tweet (one tweet out of all the documents) is less than 5% it is considered as 
#' not so useful and is given a zero probability. 
#' assignments variable/member of the lda_misinformation_high object holds the 
#' probable topic that the tweet is related to (probability using theta) within 
#' the 15 topics and assigns the corresponding labels.  
#' ---------------------------------------------------------------------------- 

# give a hard in/out assignment of topics in documents
lda_misinformation_high$assignments <- lda_misinformation_high$theta
lda_misinformation_high$assignments[lda_misinformation_high$assignments <
                                      0.05 ] <- 0
lda_misinformation_high$assignments <- 
  lda_misinformation_high$assignments / rowSums(
    lda_misinformation_high$assignments)
lda_misinformation_high$assignments[ is.na(lda_misinformation_high$assignments)
                                     ] <- 0

# Get some topic labels using n-grams from the DTM
lda_misinformation_high$labels <- LabelTopics(
  assignments = lda_misinformation_high$assignments, 
  dtm = xdtm, M = 2)

head(lda_misinformation_high$assignments)

# Number of documents in which each topic appears (Again remember, reach tweet  
# might be related to more than 1 topic)
lda_misinformation_high$num_docs <- colSums(lda_misinformation_high$assignments
                                            > 0)

# Cluster topics together in a dendrogram
# Calculate the Hellinger distance using CalcHellingerDist method and using the 
# phi vectors (or the word probabilities of the topics)
# Perform Hierarchical clustering using hclust method and the linguistic distance
# calculated using the CalcHellingerDist method. Use the "ward.D" agglomerative 
# clustering technique. 
# Limit the number of clusters to 10 instead of 15 based on the hclust you have 
# calculated earlier.
lda_misinformation_high$linguistic <- CalcHellingerDist(
  lda_misinformation_high$phi)
lda_misinformation_high$hclust <- hclust(as.dist(
  lda_misinformation_high$linguistic),"ward.D")
lda_misinformation_high$hclust$labels <- paste(lda_misinformation_high$hclust$labels, 
                                               lda_misinformation_high$labels[,1])
# given from assignment
lda_misinformation_high$hclust$clustering <- 
  cutree(lda_misinformation_high$hclust, k = 10)

# Create labels for the clusters. The code sample above created two set of 
# labels using the LabelTopics function. You could combine these two labels into 
# one as shown below: 
lda_misinformation_high$hclust$labels <- 
  paste(lda_misinformation_high$hclust$labels, 
        lda_misinformation_high$labels[ , 1])

# Plot the hclust of your model using plot function , plot the cluster dendogram
plot(lda_misinformation_high$hclust)

# Mke a summary table
lda_misinformation_high$summary <- data.frame(topic = rownames(lda_misinformation_high$phi),
                                              cluster   = lda_misinformation_high$hclust$clustering,
                                              lda_misinformation_high$labels,
                                              coherence = lda_misinformation_high$coherence,
                                              num_docs  = lda_misinformation_high$num_docs,
                                              top_terms = apply(lda_misinformation_high$top_Terms, 2, function(x){
                                                paste(x, collapse = ", ")
                                              }),
                                              stringsAsFactors = FALSE)

View(lda_misinformation_high$summary[order(lda_misinformation_high$hclust$clustering) , ])

# Repeat the same steps for the low-harm misinformation tweets

loharm <- mydata %>%
  filter(Harm == "Low Harm")

lh_tokens <- loharm %>%
  mutate(tweetId = row_number()) %>%
  unnest_tokens(word, Tweet) %>%
  anti_join(stop_words, by = "word") %>%
  count(word, tweetId)

# Lemmatize the words
lh_tokens$word <- lemmatize_words(lh_tokens$word, dictionary = lexicon::hash_lemmas)

# DTM CREATION (Document Term Matrix)
# Assign the broken out tokens to the hh_DTM variable
lh_DTM <- lh_tokens %>%
  cast_dtm(tweetId, word, n)

# Create the DTM & remove stopwords
xdtm1 <- VCorpus(VectorSource(loharm$Tweet)) %>%
  tm_map(removeWords, stopwords("en"))

# Create the DTM
xdtm1 <- CreateDtm(
  doc_vec = loharm$Tweet,
  doc_names = rownames(loharm),
  ngram_window = c(1, 2), # Bi-grams instead of only words
  stopword_vec = c(stopwords::stopwords("en"), stopwords::stopwords(source = "smart")),
  lower = TRUE,
  remove_punctuation = TRUE,
  remove_numbers = TRUE,
  stem_lemma_function = NULL,
  verbose = FALSE)

#' Fit the LDA model model for the Low-Harm tweets. Generate 15 topics and 
#' use 500 iterations while fitting the LDA topic model
#' Name the topic model you create as lda_misinformation_lo
#  Apply the model to my xdtm data (15 topics and 500 iterations)
lda_misinformation_lo <- FitLdaModel(
  xdtm1,
  k = 15,
  iterations = 500,
  burnin = 200, # Hyper - parameters
  alpha = 0.1,
  beta = 0.05,
  optimize_alpha = TRUE,
  calc_likelihood = TRUE,
  calc_coherence = TRUE,
  calc_r2 = TRUE)

#' Print the R^2 value of the model which will be there as a r2 member of 
#' Hint: You can print the r2 from lda_misinformation_high$r2 
# Display the r^2 value (higher is better, prob close to 0.1)
lda_misinformation_lo$r2

#' Plot the coherence histogram. Coherence value would have been calculated 
#' and stored in the lda_misinformation_high$coherence 
# Topic Coherence (sum of the betas of all the words in each topic?)
sort(lda_misinformation_lo$coherence)
lda_misinformation_lo$coherence
hist(lda_misinformation_lo$coherence)

#' 7. Capture the top 15 terms for each topic ,Top M (qty) terms
lda_misinformation_lo$top_Terms <- GetTopTerms(lda_misinformation_lo$phi, 
                                               M = 15, return_matrix = TRUE)
head(lda_misinformation_lo$top_Terms)

#' ----------------------------------------------------------------------------
#' Assigning the labels for each topic using the set of topic keywords is the  
#' most important task of topic modeling. The textmineR package provides a simple 
#' implementation of assigning the topic lables using the best N-grams with the  
#' highest theta value. Phi is the word probability distribution per topic and 
#' theta is the topic-level probability distribution over the documents. 
#' Here in the below code, if the probability of a topic related for a particular 
#' tweet (one tweet out of all the documents) is less than 5% it is considered as 
#' not so useful and is given a zero probability. 
#' assignments variable/member of the lda_misinformation_high object holds the 
#' probable topic that the tweet is related to (probability using theta) within 
#' the 15 topics and assigns the corresponding labels.  
#' ---------------------------------------------------------------------------- 

# give a hard in/out assignment of topics in documents
lda_misinformation_lo$assignments <- lda_misinformation_lo$theta
lda_misinformation_lo$assignments[lda_misinformation_lo$assignments < 0.05 ] <- 0
lda_misinformation_lo$assignments <- 
  lda_misinformation_lo$assignments / rowSums(lda_misinformation_lo$assignments)
lda_misinformation_lo$assignments[ is.na(lda_misinformation_lo$assignments) ] <- 0

# Get some topic labels using n-grams from the DTM
lda_misinformation_lo$labels <- LabelTopics(
  assignments = lda_misinformation_lo$assignments, 
  dtm = xdtm1, M = 2)
head(lda_misinformation_lo$assignments)

# Number of documents in which each topic appears (Again remember, each tweet  
# might be related to more than 1 topic).
lda_misinformation_lo$num_docs <- colSums(lda_misinformation_lo$assignments > 0)

# Cluster topics together in a dendrogram
# Calculate the Hellinger distance using CalcHellingerDist method and using the 
# phi vectors (or the word probabilities of the topics)
# Perform Hierarchical clustering using hclust method and the linguistic distance
# calculated using the CalcHellingerDist method. Use the "ward.D" agglomerative 
# clustering technique. 
# Limit the number of clusters to 10 instead of 15 based on the hclust you have 
# calculated earlier.

lda_misinformation_lo$linguistic <- CalcHellingerDist(lda_misinformation_lo$phi)
lda_misinformation_lo$hclust <- hclust(as.dist(lda_misinformation_lo$linguistic),"ward.D")
lda_misinformation_lo$hclust$labels <- paste(lda_misinformation_lo$hclust$labels, 
                                             lda_misinformation_lo$labels[,1])
# given from assignment
lda_misinformation_lo$hclust$clustering <- 
  cutree(lda_misinformation_lo$hclust, k = 10)

# Create labels for the clusters. The code sample above created two set of 
# labels using the LabelTopics function. You could combine these two labels into 
# one as shown below: 
lda_misinformation_lo$hclust$labels <- 
  paste(lda_misinformation_lo$hclust$labels, lda_misinformation_lo$labels[ , 1])

# Plot the lo clust of your model using plot function, cluster dendogram
plot(lda_misinformation_lo$hclust)

# Finally, make a summary table
lda_misinformation_lo$summary <- data.frame(topic = rownames(lda_misinformation_lo$phi),
                                            cluster   = lda_misinformation_lo$hclust$clustering,
                                            lda_misinformation_lo$labels,
                                            coherence = lda_misinformation_lo$coherence,
                                            num_docs  = lda_misinformation_lo$num_docs,
                                            top_terms = apply(lda_misinformation_lo$top_Terms, 2, function(x){
                                              paste(x, collapse = ", ")
                                            }),
                                            stringsAsFactors = FALSE)

# View the summary
View(lda_misinformation_lo$summary[order(lda_misinformation_lo$hclust$clustering) , ])


# -----------------------------END OF CODE-------------------------------------

