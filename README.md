# Sentiment Analysis Using Text

## Requirements
<ins>Softwares</ins>:
Python3, Jupyter notebook, VS code(or any other code editor)

<ins>Libraries</ins>:
Pandas, Numpy, Matplotlib, nltk, re, tqdm, Sklearn, gensim, pickle, tkinter, pillow(PIL)

## Analysis
<p align="justify">
This folder has the ipython notebook that covers all the data extraction, data cleaning, vectorization of text and modelling to find out the best means of vectorization and
the best model. We already have train, cross validate and test data seperately, which we will use in training, hyperparameter tuning and model selection respectively.
</p>

<p align="justify">
<ins>Data Extraction</ins>:
The data is present in txt format and the columns are seperated by "  ". Also this data is for another task that is emotion intensity prediction, but we are gonna use it for 
emotion detection so we need to make some changes like not selecting very less intense emotions, for every emotion we use intensity>0.4 to filter out the data. Finally the data 
is stored into a data frame with two columns "text" and "emotion". We have three data frames train, test and cross validate. Also we encode the class labels(angry->0, fear->1, 
joy->2, sadness->3).
</p>

<p align="justify">
<ins>Data Cleaning</ins>:
Removal of stop words, stemming, removal of tags and extra characters all is done in this step. By the end of this step we have two different types of data frames, one is for 
word 2 vec and another for bag of words. They are different because word to vec takes list of list of words as input while bag of words takes list of sentences as input.
</p>

<p align="justify">
<ins>Text Vectorization</ins>: 
We do bag of words and word2vec vectorization of text, using different vectorization for different models. Normalization of both vectorized features is also done in this very 
step. We now have our whole data in form of feature matrix and class label array that are numpy arrays and can be directly fed to the models.
</p>
 
<p align="justify">
<ins>Modelling</ins>:
We have used KNN(word2vec), Logistic Regression(bag of words), Naive Bayes(bag of words), Random forest(word2vec), Linear SVM(word2vec), Linear SVM(bag of words) and stacking
of the three models(Logistic Regression, Naive Bayes and SVM). LinearSVM with bag of words gace the best accuracy and confusion matrix so we will use it in the final model.
</p>

## Final Model
<p align="justify">
This folder has the py file where final model is trained on all of the data(train, cross validate and test) with the best model and hyperparameter obtained after analysis from 
the ipython notebook. We user Linear SVM with bag of words vectorization and C=10 as the Analysis indicated it to be the best performer. It dumps the model into pickle file in 
the GUI folder which can now be used in production directly in our tkinter GUI. 
</p>

<p align="justify">
<ins>MainPickle.pkl</ins>:
We dump the bag of words model, main model and data all into a single pickle file as a tuple object, we need all of them to allow for dynamic learning of model during the time 
user uses the model.
</p>

<ins>StopStem.pkl</ins>:
We also dump our set of stopwords and stemmer so we do not need nltk library in the main interface. 

## GUI
<p align="justify">
This folder has has our main interface coded into the GUI window python file. We load the pickle objects into this file and use tkinter for the front interface. The interface 
can be used to simply enter text and then predict, it asks for a confirmation if the prediction was right, if not, it asks for the correct prediction and on telling it that
it saves it in a list. When user presses the save button on main window, model is fitted to all the entries of that session along with previous data and the data is also updated
then we restart the application and now it remembers the words told to it previously.
</p>

<ins>Background</ins>:
The background image for gui is taken from Fone Walls images.<br>
<ins>Link</ins>: https://www.fonewalls.com/720x1280-wallpapers/720x1280-background-hd-wallpaper-375/

## Raw Data 
The data is already partitioned into train, test and cv. It is present as text file.

<p align="justify">
<ins>Referred from</ins>:
WASSA-2017 Shared Task on Emotion Intensity. Saif M. Mohammad and Felipe Bravo-Marquez. In Proceedings of the EMNLP 2017 Workshop on Computational Approaches to Subjectivity, 
Sentiment and Social Media(WASSA), September 2017, Copenhagen, Denmark, BibTex. Link: http://saifmohammad.com/WebPages/EmotionIntensity-SharedTask.html
</p>

## Runnning and Using Final APP
<p align="justify">
To actually run the application, you will need all the files in GUI folder and the libraries mentioned in the requirements section, then you can run the GUIWindow.py file in any
python interpreter and you will be able to use the application. The app window has a text field and two buttons, you can enter your text in the text field and press ok to get a 
prediction, the program will ask you for a confimation if the prediction is correct, if it is wrong you may press no and this will redirect to a window where you can enter the 
correct answer, then that answer will be recorded and you will be returned to main window. Just before closing the app you have to click on save to train your model on all the new data entered by you and whenever you restart it will remember.
</p>
