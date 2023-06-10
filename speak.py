import pyttsx3 #pip install pyttsx3

#computer starts speaking


def Say(Text):
    engine = pyttsx3.init("sapi5") #microsoft sapi files
    voices = engine.getProperty("voices")
    engine.setProperty("voices",voices[0].id)
    engine.setProperty("rate",170) #speed of talking = 1x = 200
    print("     ")
    print(f"A|I|: {Text}") #printing text
    engine.say(text=Text)
    engine.runAndWait()
    print("     ")

