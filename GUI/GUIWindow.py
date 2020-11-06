from tkinter import Tk,Label,Frame,Entry,Button,StringVar,Toplevel
from tkinter.ttk import Combobox
from tkinter import messagebox
from PIL import Image,ImageTk

import re 

from sklearn.preprocessing import normalize 
import pickle
import pandas

#Load pickle files
flag=0 
try:
    with open("MainPickle.pkl","rb") as file1:
        Model,bow_model,OldData =pickle.load(file1)
except:
    with open("GUI/MainPickle.pkl","rb") as file1:
        Model,bow_model,OldData =pickle.load(file1)
    flag=1

try:
    with open("StopStem.pkl","rb") as file2:
        stop,sno =pickle.load(file2)
except:
    with open("GUI/StopStem.pkl","rb") as file2:
        stop,sno =pickle.load(file2)


#Some Initialization
newData = pandas.DataFrame()
user_entries = []
user_emotions = []


#Function for prediction
def predict():
    global user_emotions, user_entries

    text = user_text.get()
    
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

    X = normalize(bow_model.transform([filtered_text])).tocsr()
    emotionDict = {0:"anger", 1:"fear", 2:"joy", 3:"sad", "anger":0, "fear":1, "joy":2, "sad":3}
    Y = Model.predict(X)
    Y = Y[0]
    res = emotionDict[Y]
    confirm = messagebox.askquestion(f"{res}",f"Do you feel {res}?")
    if(confirm=="yes"):
        return
    else:
        def commit():
            user_entries.append(filtered_text)
            user_emotions.append(emotionDict[emotion.get()])
            messagebox.showinfo("Okay","I will take care from next time after you press save")
            childWindow.destroy()

        childWindow = Toplevel()
        childWindow.geometry("150x150")
        label = Label(childWindow, text="What do you feel?")
        label.pack()
        emotion = StringVar()
        combobox = Combobox(childWindow, textvariable=emotion, values=["anger","fear","joy","sad"])
        combobox.pack()
        btn = Button(childWindow, text="Ok", command=commit)
        btn.pack()
        childWindow.mainloop()


#Function for fitting to new data   
def adaptNewData():
    global user_entries, user_emotions, newData
    if(len(user_entries)==0):
        return
    newData["text"] = user_entries
    newData["emotion"] = user_emotions
    Data = pandas.concat([OldData, newData], ignore_index=True)
    bow_model.fit(Data["text"])
    X = normalize(bow_model.transform(Data["text"])).tocsr()
    Y = Data["emotion"].to_numpy()
    Model.fit(X,Y)
    Tup_Obj = (Model, bow_model, Data)
    if(flag==0):
        with open("MainPickle.pkl","wb") as file1:
            pickle.dump(Tup_Obj, file1)
    else:
        with open("GUI/MainPickle.pkl","wb") as file1:
            pickle.dump(Tup_Obj, file1)
    print("Saved new Data")
    messagebox.showinfo("Done","Restart for changes to take effect!")
    window.destroy()


#GUI window
window = Tk()

#Window basics
window.title("Sentiment Analysis")
window.geometry("500x701")

#Background
try:
    bgImg = Image.open("background.jpg")
except:
    bgImg = Image.open("GUI/background.jpg")

Img = ImageTk.PhotoImage(bgImg)
Background = Label(window, image=Img)
Background.place(x=0,y=0,relwidth=1,relheight=1)

#Heading
headingFrame = Frame(window,bg="SlateBlue4",bd=5)
headingFrame.place(relx=0.2,rely=0.1,relwidth=0.6,relheight=0.16)
headingLabel = Label(headingFrame, text="Emotion Detection", bg='SlateBlue3', fg='black', font=('Cailibri',20))
headingLabel.place(relx=0,rely=0, relwidth=1, relheight=1)

#Variable
user_text = StringVar()

#Entry
text = Entry(window, textvariable=user_text, bg="white", fg="black", font=("Calibri",14))
text.place(relx=0.1, rely=0.4, relwidth=0.8, relheight=0.05)

submit = Button(window, text="Ok", font=("Calibri",16), command=predict)
submit.place(relx=0.300, rely=0.50, relwidth=0.2, relheight=0.05)

saveAll = Button(window, text="Save", font=("Calibri",16), command=adaptNewData)
saveAll.place(relx=0.500, rely=0.50, relwidth=0.2, relheight=0.05)

#main loop
window.mainloop()
