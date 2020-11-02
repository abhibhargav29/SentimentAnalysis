# Sentiment Analysis using Text

## SentimentAnalysis.py
The sentiment Analysis python notebook covers all the data extraction, data cleaning, vectorization of text and modelling to find out the best means of vectorization and
the best model. We already have train, cross validate and test data seperately, which we will use in training, hyperparameter tuning and model selection respectively.

### Data Extraction: 
The data is present in txt format and the columns are seperated by "  ". Also this data is for another task that is emotion intensity prediction, but we
are gonna use it for emotion detection so we need to make some changes like not selecting very less intense emotions, for every emotion we use different intensity to filter out 
based on data(This is done by manually looking at data). Finally the data is stored into a data frame with two columns "text" and "emotion". We have three data frames train, test 
and cross validate. Also we encode the class labels(angry->0, fear->1, joy->2, sadness->3).

### Data Cleaning: 
Removal of stop words, stemming, removal of tags and extra characters all is done in this step. By the end of this step we have two different types of data frames, 
one is for word 2 vec and another for bag of words. They are different because word to vec takes list of list of words as input while bag of words takes list of sentences as 
input.

### Text Vectorization: 
We do bag of words and word2vec vectorization of text. Normalization of both vectorized features is also done in this very step. We now have our whole data
in form of feature matrix and class label array that are numpy arrays and can be directly fed to the models.

### Modelling: 
We have used KNN(word2vec), Logistic Regression(bag of words), Naive Bayes(bag of words), Random forest(word2vec), Linear SVM(word2vec) and Linear SVM(bag of words). 


### Data: 
WASSA-2017 Shared Task on Emotion Intensity. Saif M. Mohammad and Felipe Bravo-Marquez. In Proceedings of the EMNLP 2017 Workshop on Computational Approaches to          
Subjectivity, Sentiment, and Social Media (WASSA), September 2017, Copenhagen, Denmark, BibTex
Link - http://saifmohammad.com/WebPages/EmotionIntensity-SharedTask.html
