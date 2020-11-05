import pandas as pd
import numpy as np

import re
import nltk
from nltk.corpus import stopwords
nltk.download("stopwords")

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score


#Train Data
trainDf =  pd.read_csv('Train.txt', sep='	', names=["id","text","emotion","intensity"], engine='python')
trainDf=trainDf.drop("id",axis=1)

AngerDf = trainDf[trainDf["emotion"]=="anger"].drop("emotion",axis=1)
AngerDf = AngerDf[AngerDf["intensity"]>=0.4]
AngerDf["emotion"] = 0

FearDf = trainDf[trainDf["emotion"]=="fear"].drop("emotion",axis=1)
FearDf = FearDf[FearDf["intensity"]>=0.4]
FearDf["emotion"] = 1

JoyDf = trainDf[trainDf["emotion"]=="joy"].drop("emotion",axis=1)
JoyDf = JoyDf[JoyDf["intensity"]>=0.4]
JoyDf["emotion"] = 2

SadDf = trainDf[trainDf["emotion"]=="sadness"].drop("emotion",axis=1)
SadDf = SadDf[SadDf["intensity"]>=0.4]
SadDf["emotion"] = 3

trainDf = pd.concat([AngerDf,FearDf,JoyDf,SadDf],ignore_index=True)
print()
print("Shape of train dataset:",trainDf.shape)

#Cross Validation Data
crossValDf =  pd.read_csv('CrossValidate.txt', sep='	', names=["id","text","emotion","intensity"], engine='python')
crossValDf=crossValDf.drop("id",axis=1)

AngerDf = crossValDf[crossValDf["emotion"]=="anger"].drop("emotion",axis=1)
AngerDf = AngerDf[AngerDf["intensity"]>0.4]
AngerDf["emotion"] = 0

FearDf = crossValDf[crossValDf["emotion"]=="fear"].drop("emotion",axis=1)
FearDf = FearDf[FearDf["intensity"]>=0.4]
FearDf["emotion"] = 1

JoyDf = crossValDf[crossValDf["emotion"]=="joy"].drop("emotion",axis=1)
JoyDf = JoyDf[JoyDf["intensity"]>=0.4]
JoyDf["emotion"] = 2

SadDf = crossValDf[crossValDf["emotion"]=="sadness"].drop("emotion",axis=1)
SadDf = SadDf[SadDf["intensity"]>=0.4]
SadDf["emotion"] = 3

crossValDf = pd.concat([AngerDf,FearDf,JoyDf,SadDf],ignore_index=True)
print("Shape of CV data:",crossValDf.shape)

#Test Data
testDf =  pd.read_csv('Test.txt', sep='	', names=["id","text","emotion","intensity"], engine='python')
testDf=testDf.drop("id",axis=1)

AngerDf = testDf[testDf["emotion"]=="anger"].drop("emotion",axis=1)
AngerDf = AngerDf[AngerDf["intensity"]>=0.4]
AngerDf["emotion"] = 0

FearDf = testDf[testDf["emotion"]=="fear"].drop("emotion",axis=1)
FearDf = FearDf[FearDf["intensity"]>=0.4]
FearDf["emotion"] = 1

JoyDf = testDf[testDf["emotion"]=="joy"].drop("emotion",axis=1)
JoyDf = JoyDf[JoyDf["intensity"]>=0.4]
JoyDf["emotion"] = 2

SadDf = testDf[testDf["emotion"]=="sadness"].drop("emotion",axis=1)
SadDf = SadDf[SadDf["intensity"]>=0.4]
SadDf["emotion"] = 3

testDf = pd.concat([AngerDf,FearDf,JoyDf,SadDf],ignore_index=True)
print("Shape of Test data:",testDf.shape)

#Merge
Data = pd.concat([trainDf, crossValDf, testDf], ignore_index=True).drop("intensity",axis=1)
print(Data.head(10))
print()
print("Shape of final combined data:",Data.shape)
print()


#Text Cleaning
stop=set(stopwords.words("english"))
stop.discard("not")
stop.discard("no")
sno = nltk.stem.SnowballStemmer("english")

Rawtext=Data["text"]
cleaned_text=[]
for line in Rawtext:
    #Removing tags(ex-@abhishek is a name and not needed)
    tags = re.compile("^@[a-zA-Z_]*")
    line = re.sub(tags," ",line)
    #Replacing # and * with a ""
    hashtags = re.compile("#|\*")
    line = re.sub(hashtags,"",line)
    #Replacing all other characters with a space
    extraCharacters = re.compile("[^a-zA-Z]")
    line = re.sub(extraCharacters," ",line)

    #Convert to lower case, stemming, stopword removal
    filtered_words=""
    for word in line.split():
        word=word.lower()
        if(word not in stop):
            word = sno.stem(word)
            filtered_words+=" "+word
    cleaned_text.append(filtered_words)
            
CleanedData = pd.DataFrame(data=cleaned_text,columns=["text"])
CleanedData["emotion"] = Data["emotion"]
print("Cleaned Data: ")
print(CleanedData.head(10))
print()


#Vectorization
bow_model = CountVectorizer(ngram_range=(1,2))
bow_model.fit(CleanedData["text"])

X = normalize(bow_model.transform(CleanedData["text"])).tocsr()
y = CleanedData["emotion"].to_numpy()

print("Final shape of X and y for training:",X.shape,y.shape)
print()

#Model
finalModel = LinearSVC(C = 10, dual=False)
finalModel.fit(X, y)

#New Prediction
while(True):
    user_text = input("Enter text to predict(Enter 'e' to exit): ")
    if(user_text=="e"):
        print("Bye")
        break
    tags = re.compile("^@[a-zA-Z_]*")
    user_text = re.sub(tags," ",user_text)
    hashtags = re.compile("#|\*")
    user_text = re.sub(hashtags,"",user_text)
    extraCharacters = re.compile("[^a-zA-Z]")
    user_text = re.sub(extraCharacters," ",user_text)
    
    filtered_text=""
    for word in user_text.split():
        word=word.lower()
        if(word not in stop):
            word = sno.stem(word)
            filtered_text+=" "+word

    X = normalize(bow_model.transform([filtered_text])).tocsr()
    Y = finalModel.predict(X)
    Y = Y[0]
    if(Y==0):
        print("Anger")
    elif(Y==1):
        print("Fear")
    elif(Y==2):
        print("Joy")
    else:
        print("Sadness")
