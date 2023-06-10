#Features--- non-input functions, input functions


#non-input functions
import datetime
from speak import Say
import wikipedia
import pywhatkit

def Time():
    time=datetime.datetime.now().strftime("%H:%M")
    Say(time)

def Date():
    date = datetime.date.today()
    Say(date)

def Day():
    day = datetime.datetime.now().strftime("%A")
    Say(day)

def nonInputFunc(query):

    query = str(query)
    if "time" in query:
        Time()

    elif "date" in query:
        Date()
    
    elif "day" in query:
        Day()


#input functions

# def wikipedia(tag,query):
#     name = str(query).replace("","")

#     import wikipedia
#     result = wikipedia.summary(name)
#     Say(result)


def inputFunc(tag,query):
    if "wikipedia" in tag:
        name = str(query).replace("who is","").replace("what is","").replace("how","").replace("where is","").replace("wikipedia","")
        result = wikipedia.summary(name)
        Say(result)
    elif "google" in tag:
        query= str(query).replace("google","")
        query= query.replace("search","")
        pywhatkit.search(query)

    




