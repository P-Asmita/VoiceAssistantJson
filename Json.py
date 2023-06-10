import random
import json
import torch
from think import neuralNetwork 
from neuralNet import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open("emotions.json",'r') as json_data:
    emotions = json.load(json_data)

FILE = "trainingData.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = neuralNetwork(input_size,hidden_size,output_size).to(device)
model.load_state_dict(model_state)
model.eval()

#---------------------------------------------
Name = "Json"
from listen import listen
from speak import Say
from task import nonInputFunc
from task import inputFunc

def Main():
    sentence = listen()
    out = str(sentence)

    if sentence == "bye":
        exit()

    sentence = tokenize(sentence)
    X = bag_of_words(sentence,all_words)
    X = X.reshape(1,X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _ , predicted = torch.max(output,dim=1)

    tag = tags[predicted.item()]

    probabs = torch.softmax(output,dim=1)

    probab = probabs[0][predicted.item()]

    if probab.item() > 0.75:
        for emotion in emotions['emotions']:
            if tag == emotion["tag"]:
                reply = random.choice(emotion["responses"])

                if "time" in reply:
                    nonInputFunc(reply)
                elif "date" in reply:
                    nonInputFunc(reply)
                elif "day" in reply:
                    nonInputFunc(reply)

                elif "wikipedia" in reply:
                    inputFunc(reply,out)
                
                elif "google" in reply:
                    inputFunc(reply,out)

                else:
                    Say(reply)



while True:
    Main()





