# Sentiment Analysis using Text

The sentiment Analysis python notebook covers all the data extraction, data cleaning, vectorization of text and modelling to find out the best means of vectorization and
the best model. We already have train, cross validate and test data seperately, which we will use in training, hyperparameter tuning and model selection respectively.

Data Extraction: The data is present in txt format and the columns are seperated by "  ". Also this data is for another task that is emotion intensity prediction, but we
are gonna use it for emotion detection so we need to make some changes like not selecting very less intense emotions, for every emotion we use different intensity to filter out 
based on data(This is done by manually looking at data). Finally the data is stored into a data frame with two columns "text" and "emotion". We have three data frames train, test 
and cross validate.

Data Cleaning: 

Data: WASSA-2017 Shared Task on Emotion Intensity. Saif M. Mohammad and Felipe Bravo-Marquez. In Proceedings of the EMNLP 2017 Workshop on Computational Approaches to          
      Subjectivity, Sentiment, and Social Media (WASSA), September 2017, Copenhagen, Denmark, BibTex
      Link - http://saifmohammad.com/WebPages/EmotionIntensity-SharedTask.html
