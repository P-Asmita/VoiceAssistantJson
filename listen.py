#Recognize what i say and print the same
#pip install piaudio

import speech_recognition as sr #pip install speech_recognition

def listen():
    ai_listening=sr.Recognizer()

    with sr.Microphone() as source:
        #input audio
        print("Listening to what you are saying")
        ai_listening.pause_threshold = 1
        audio = ai_listening.listen(source,0,4) #after every 2 secs it'll start recognizing

    try:
        print("Recognizing your speech")
        query = ai_listening.recognize_google(audio,language="en-in")
        print(f"You said: {query}\n")

    except:
        return ""
    
    query=str(query)
    return query.lower()

listen()


