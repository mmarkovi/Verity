#Written by Mia Markovic (mmarkovi, 39425669) using base code provided for assignment 2 for CS 175
# and with help from the article at 
# https://medium.com/analytics-vidhya/part-1-sentiment-analysis-in-pytorch-82b35edb40b8
 
# ---------------------------------------------------------------------------------------
# CS 175, WINTER 2021: VERITY INITIAL SIMPLE MODEL
#
# ---------------------------------------------------------------------------------------


from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np

import sys
sys.path.insert(0, '../preprocessing/') #need this in order to get to the other file in other directory

#can comment out the ones you aren't using to save a little bit of time
# from covidPreprocess import getCoronaVocabulary, getCoronaText
# from liarPreprocess import getLiarVocabulary, getLiarText
from fnnPreprocess import getFNNVocabulary, getFNNText


class SimpleNeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNeuralNet, self).__init__()
		#Written based off of the tutorial at
		#https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/01-basics/feedforward_neural_network/main.py#L37-L49
        self.hidden1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.hOutput1 = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim = 0)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.hidden1.weight.data.uniform_(-initrange, initrange)
        self.relu.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, x):
        out = self.hidden1(x)
        out = self.relu(out)
        out = self.hOutput1(out)
        out = self.softmax(out)
        return out
    

def trainAndTestSimpleModel():
    '''
    gets around 63-71% for corona and Liar datasets, around 80-83% on FNN
    
    used this article for help in writing the tensor parts of code so it works with the model
    https://medium.com/analytics-vidhya/part-1-sentiment-analysis-in-pytorch-82b35edb40b8
    '''
    # X,Y = getCoronaText() #this function will give us the text array (not document term matrix) and Y
    # X_train,Y_train, vectorizer_train = getCoronaVocabulary(True)
    # X,Y = getLiarText()
    # X_train,Y_train, vectorizer_train = getLiarVocabulary(True)
    X,Y = getFNNText()
    X_train,Y_train, vectorizer_train = getFNNVocabulary(True)
    
    #transform our testing dataset to match the vocabulary for the training dataset
    #transform will return the document-term matrix for X based on training dataset
    x_test = vectorizer_train.transform(X)
    
    #sample test on logistic classifier
    '''classifier = LogisticRegression()
    classifier.fit(X_train,Y_train)
    score = classifier.score(x_test,Y)
    print(score)'''
    
    vocabsize = X_train.shape[1]
    
    
    #transform our training and test data into tensors for the classifier to learn off of
    X_tensor = torch.from_numpy(X_train.todense()).float()
    Y_tensor = torch.from_numpy(np.array(Y_train))
    
    X_test_tensor = torch.from_numpy(x_test.todense()).float()
    Y_test_tensor = torch.from_numpy(np.array(Y))
    
    device = torch.device('cpu')
    #use TensorDataset to be able to use our DataLoader
    train_data = torch.utils.data.TensorDataset(X_tensor, Y_tensor)
    train_loader = torch.utils.data.DataLoader(train_data,batch_size=16, shuffle=True)
    
    test_data = torch.utils.data.TensorDataset(X_test_tensor, Y_test_tensor)
    test_loader = torch.utils.data.DataLoader(test_data,batch_size=16, shuffle=True)
    
    
    num_epochs = 5
    learning_rate = 0.001
    #initialize our model
    model = SimpleNeuralNet(1, 200, 2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    
    
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (x_batch, labels) in enumerate(train_loader):
    
            # Forward pass
            # The forward process computes the loss of each iteration on each sample
            model.train()
            y_pred = model(x_batch)
            #need to transform labels to long datatype using .long() or it complains it's an int
            loss = criterion(y_pred, labels.long())
    
            # Backward pass, using the optimizer to update the parameters
            optimizer.zero_grad()
            loss.backward()    #compute gradients
            optimizer.step()   #initiate gradient descent
    
     
            # Below, an epoch corresponds to one pass through all of the samples.
            # Each training step corresponds to a parameter update using 
            # a gradient computed on a minibatch of 100 samples 
            if (i + 1) % 5 == 0: 
                #leaving it on 5 for corona dataset, probably want to change to % 50 or % 100
                # for the other datasets so don't get spammed 
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
    
    # Test the model
    # In the test phase, we don't need to compute gradients (the model has already been learned)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))

if __name__ == "__main__":
    trainAndTestSimpleModel()

