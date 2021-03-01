#Written by Mia Markovic (mmarkovi, 39425669) with some help from the article at 
# https://stackabuse.com/scikit-learn-save-and-restore-models/
 
# ---------------------------------------------------------------------------------------
# CS 175, WINTER 2021: VERITY SAVING AND LOADING MODEL
#
# ---------------------------------------------------------------------------------------
from __future__ import unicode_literals, print_function, division
import torch
import numpy as np
import pickle
import torch
import torch.nn as nn

import sys

sys.path.insert(0, '../preprocessing/') #need this in order to get to the other file in other directory

#can comment out the ones you aren't using to save a little bit of time
from covidPreprocess import getCoronaText, get_whole_Corona_dataset
from liarPreprocess import getLiarText
from fnnPreprocess import getFNNText

from preprocessingFunctions import getTermMatrixTestData

class SimpleNeuralNet(nn.Module):

	def __init__(self, input_size, hidden_size, num_classes):
		super(SimpleNeuralNet, self).__init__()
		#Written based off of the tutorial at
		#https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/01-basics/feedforward_neural_network/main.py#L37-L49
		self.hidden1 = nn.Linear(input_size, hidden_size) 
		self.relu = nn.ReLU()   
		self.hOutput1 = nn.Linear(hidden_size, num_classes)  
		self.softmax = nn.Softmax(dim = 0)

	def forward(self, x):
		out = self.hidden1(x)
		out = self.relu(out)
		out = self.hOutput1(out)
		out = self.softmax(out)
		return out

def saveModel(model, vectorizer):
    # Save to file in the current working directory
    pkl_filename = "pickle_vec.pkl"
    torch.save(model, 'saved_model')

    with open(pkl_filename, 'wb') as file:
        pickle.dump(vectorizer, file)
    
def loadModel():
    # Load from file
    pkl_filename = "pickle_vec.pkl"
    model_filename = "saved_model"
    with open(pkl_filename, 'rb') as file:
        vec = pickle.load(file)

    model = torch.load(model_filename)
    return model, vec

#from simpleModel import SimpleNeuralNet
# ^^^^^ must include this line when importing these functions below
#CAN'T TEST THESE IN THIS FILE SINCE YOU NEED SIMPLENEURALNET AND THAT CAUSES CIRCULAR IMPORT

def testDataset(dataset: str = 'corona'):
    model, vec = loadModel()
    #get dataset we want to use
    if dataset == 'fnn':
        X,Ytest = getFNNText()
    elif dataset == 'liar':
        X,Ytest = getLiarText()
    else:
        #dataset == 'corona':
        X,Ytest = getCoronaText() #this function will give us the text array (not document term matrix) and Y
    Xtest = vec.transform(X)
    X_test_tensor = torch.from_numpy(Xtest.todense()).float()
    Y_test_tensor = torch.from_numpy(np.array(Ytest))
    test_data = torch.utils.data.TensorDataset(X_test_tensor, Y_test_tensor)
    test_loader = torch.utils.data.DataLoader(test_data,batch_size=16, shuffle=True)
    
    # Calculate the accuracy score and predict target values
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))

def testText(text:str, label):
    model, vec = loadModel()
    #transform the data passed in
    Xtest = getTermMatrixTestData(text, vec)
    X_test_tensor = torch.from_numpy(Xtest.todense()).float()
    #test against model
    output = model(X_test_tensor)
    _, predicted = torch.max(output.data, 1)
    Ytest = predicted.numpy()[0]
    if Ytest == label:
        print("Correctly Evaluated label as", "fake" if label == 0 else "true")
    else:
        print("Incorrectly labeled", "fake" if label == 0 else "true", "article as", "fake" if Ytest == 0 else "true")
    return Ytest
    

def train_and_save_simple_model(num_epochs = 5, learning_rate = 0.001, print_epoch_mod = 5, DEBUG_MODE = False):
    '''
    trains with the selected dataset using simple neural network
    saves the model using pickle
    
    used this article for help in writing the tensor parts of code so it works with the model
    https://medium.com/analytics-vidhya/part-1-sentiment-analysis-in-pytorch-82b35edb40b8
    '''

    torch.manual_seed(1)
    
    #sample test on logistic classifier
    '''classifier = LogisticRegression()
    classifier.fit(X_train,Y_train)
    score = classifier.score(x_test,Y)
    print(score)'''

    device = torch.device("cpu")

    X, Y, vectorizer_train = get_whole_Corona_dataset()
    X_train = X.todense()
    Y_train = np.array(Y)

    X_train_tensor = torch.from_numpy(X_train).float()
    Y_train_tensor = torch.from_numpy(Y_train)

    train_data = torch.utils.data.TensorDataset(X_train_tensor, Y_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=False)
    vocabsize = X_train.shape[1]
    
    #initialize our model
    model = SimpleNeuralNet(1, 200, 2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    
    model.train()

    for epoch in range(num_epochs):
        for x_batch, labels in train_loader:
            # data = torch.from_numpy(x_batch.numpy().reshape(vocabsize)).float()
            # print(labels, labels.shape)
            # labels = torch.from_numpy(labels.numpy().reshape(1))

            # print(data, data.shape)
            # print(labels, labels.shape)
            # Forward pass
            # The forward process computes the loss of each iteration on each sample
            y_pred = model(x_batch)

            # print(y_pred)
            #need to transform labels to long datatype using .long() or it complains it's an int
            loss = criterion(y_pred, labels.long())
    
            # Backward pass, using the optimizer to update the parameters
            optimizer.zero_grad()
            loss.backward()    #compute gradients
            optimizer.step()   #initiate gradient descent

    model.eval()
    saveModel(model, vectorizer_train)

    x_sample = torch.from_numpy(X_train[:1, :]).float()
    y_pred = model(x_sample)

    print(y_pred)

    return model, vectorizer_train

if __name__ == '__main__':
    train_and_save_simple_model()