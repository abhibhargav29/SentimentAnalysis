import pandas as pd
import numpy as np

import re
import nltk
from nltk.corpus import stopwords
nltk.download("stopwords")

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

from sklearn.svm import LinearSVC

import pickle

#Train Data
#Try statement to handle if the parent directory is the working folder in vs code
flag=0
try:
    trainDf =  pd.read_csv('../Data/Train.txt', sep='	', names=["id","text","emotion","intensity"], engine='python')
    trainDf=trainDf.drop("id",axis=1)
except:
    trainDf =  pd.read_csv('Data/Train.txt', sep='	', names=["id","text","emotion","intensity"], engine='python')
    trainDf=trainDf.drop("id",axis=1)
    flag=1

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
#Try statement to handle if the parent directory is the working folder in vs code
try:
    crossValDf =  pd.read_csv('../Data/CrossValidate.txt', sep='	', names=["id","text","emotion","intensity"], engine='python')
    crossValDf=crossValDf.drop("id",axis=1)
except:
    crossValDf =  pd.read_csv('Data/CrossValidate.txt', sep='	', names=["id","text","emotion","intensity"], engine='python')
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
#Try statement to handle if the parent directory is the working folder in vs code
try:
    testDf =  pd.read_csv('../Data/Test.txt', sep='	', names=["id","text","emotion","intensity"], engine='python')
    testDf=testDf.drop("id",axis=1)
except:
    testDf =  pd.read_csv('Data/Test.txt', sep='	', names=["id","text","emotion","intensity"], engine='python')
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

#Dump Models, Data, etc
path = ""
if(flag==0):
    path1 = "../GUI/MainPickle.pkl"
    path2 = "../GUI/StopStem.pkl"
else:
    path1 = "GUI/MainPickle.pkl"
    path2 = "GUI/StopStem.pkl"

Tuple_Obj1 = (finalModel, bow_model, CleanedData)
Tuple_Obj2 = (stop, sno)

with open(path1,"wb") as file1:
    pickle.dump(Tuple_Obj1, file1)

with open(path2,"wb") as file2:
    pickle.dump(Tuple_Obj2, file2)
