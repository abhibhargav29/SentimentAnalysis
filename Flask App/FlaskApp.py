from flask import Flask, render_template, request, redirect
import pickle
import re 
from sklearn.preprocessing import normalize 
import pandas
import sys

#Defibe flask app
app = Flask(__name__)

#This try except is for the case you open the whole project folder is VS code 
flag=0 
try:
    with open("MainPickle.pkl","rb") as file1:
        Model,bow_model,OldData = pickle.load(file1)
    with open("StopStem.pkl","rb") as file2:
        stop,sno =pickle.load(file2)
except:
    with open("Flask App/MainPickle.pkl","rb") as file1:
        Model,bow_model,OldData = pickle.load(file1)
    with open("Flask App/StopStem.pkl","rb") as file2:
        stop,sno =pickle.load(file2)
    flag=1
        
user_entry = None
emotionDict = {0:"anger", 1:"fear", 2:"joy", 3:"sad", "anger":0, "fear":1, "joy":2, "sad":3}
    
#Main Window
@app.route("/", methods=["GET","POST"])
def mainWin():
    global Model,bow_model
    global sno, stop
    global user_entry, emotionDict
    res=""

    if(request.method == "POST"):
        text = request.form["text"]
        filtered_text = clean(text, sno, stop)
        X = normalize(bow_model.transform([filtered_text])).tocsr()
        Y = Model.predict(X)[0]
        res = emotionDict[Y]
        user_entry = filtered_text

    return render_template(("main.html"), result=res)

#Edit Window
@app.route("/edit", methods=["GET","POST"])
def editAns():
    global Model,bow_model,OldData,flag
    global sno, stop
    global user_entry, emotionDict

    if(request.method=="POST"):
        ans = request.form["Sentiment"]
        NewData = pandas.DataFrame()
        NewData["text"] = [user_entry]
        NewData["emotion"] = [emotionDict[ans]] 
        OldData = pandas.concat([OldData, NewData], ignore_index=True)
        bow_model.fit(OldData["text"])
        X = normalize(bow_model.transform(OldData["text"])).tocsr()
        Y = OldData["emotion"].to_numpy()
        Model.fit(X,Y)
        Tup_Obj = (Model, bow_model, OldData)
        if(flag==0):
            with open("MainPickle.pkl","wb") as file1:
                pickle.dump(Tup_Obj, file1)
        else:
            with open("Flask App/MainPickle.pkl","wb") as file1:
                pickle.dump(Tup_Obj, file1)
        return redirect("/")

    return render_template(("edit.html"))

#Cleaning function
def clean(text, sno, stop):
    tags = re.compile("^@[a-zA-Z_]*")
    text = re.sub(tags," ",text)

    hashtags = re.compile("#|\*")
    text = re.sub(hashtags,"",text)

    extraCharacters = re.compile("[^a-zA-Z]")
    text = re.sub(extraCharacters," ",text)

    filtered_text=""
    for word in text.split():
        word=word.lower()
        if(word not in stop):
            word = sno.stem(word)
            filtered_text+=" "+word
    
    return filtered_text

#driver code
if __name__=="__main__":
    app.run(debug=True)
