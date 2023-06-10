import numpy as np
import json
import torch  #deeplearning ml library
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from neuralNet import bag_of_words, tokenize, stem
from think import neuralNetwork


with open('emotions.json','r') as f:
    emotions= json.load(f)

all_words = []
tags = []
xy = []

#loading all tags 
for emotion in emotions['emotions']:
    tag = emotion['tag']
    tags.append(tag)

    #loading all patters wrt tags
    for pattern in emotion['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w,tag))  #x=w, y=tag  (pattern, tag)

ignore_words=[',','?','/','.','!']

all_words = [stem(w) for w in all_words if w not in ignore_words]

all_words = sorted(set(all_words))

tags = sorted(set(tags))


#using machine learning to train model for any different input

xTrain = []
yTrain = []

for (pattern_sentence,tag) in xy:
    bag = bag_of_words(pattern_sentence,all_words)
    xTrain.append(bag)


    label = tags.index(tag)
    yTrain.append(label)

xTrain = np.array(xTrain)
yTrain = np.array(yTrain)

#model specification
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(xTrain[0])
hidden_size = 8
output_size = len(tags)

print("Training the model..")


class chatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(xTrain)
        self.xData = xTrain
        self.yData = yTrain

    def __getitem__(self,index):
        return self.xData[index],self.yData[index]
    

    def __len__(self):
        return self.n_samples


dataset = chatDataset()

#load the trained model
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

#torch works in cuda framework, if cuda available else works in cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#work your nn model and send to device
model = neuralNetwork(input_size,hidden_size,output_size).to(device)

#loss function
criteria = nn.CrossEntropyLoss()

#optimize and input learning rate of model
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)


for epoch in range(num_epochs):
    for(words,labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)

        outputs = model(words)
        loss = criteria(outputs,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}],Loss: {loss.item():.4f}')

print(f'Final loss: {loss.item():.4f}')

data = {
    "model_state" : model.state_dict(),
    "input_size" : input_size,
    "hidden_size" : hidden_size,
    "output_size" : output_size,
    "all_words" : all_words,
    "tags" : tags
}

FILE = "trainingData.pth"
torch.save(data,FILE)

print(f"Training complete, file saved to {FILE}")










