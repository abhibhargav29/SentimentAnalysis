from flask import Flask, render_template, request, redirect
import pickle
import re 
from sklearn.preprocessing import normalize 
import pandas
import sys

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
        user_entry = X
        Y = Model.predict(X)[0]
        res = emotionDict[Y]
        user_entry = filtered_text

    return render_template(("main.html"), result=res)

@app.route("/edit", methods=["GET","POST"])
def editAns():
    global Model,bow_model,OldData
    global sno, stop
    global user_entry, emotionDict

    if(request.method=="POST"):
        ans = request.form["Sentiment"]
        NewData = pandas.DataFrame()
        NewData["text"] = [user_entry]
        NewData["emotion"] = [emotionDict[ans]] 
        Data = pandas.concat([OldData, NewData], ignore_index=True)
        OldData = Data
        print(Data)
        print(OldData)
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


if __name__=="__main__":
    app = Flask(__name__)
    
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
    
    app.run(debug=True)
