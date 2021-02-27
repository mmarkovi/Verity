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


import sys
#sys.path.insert(0, '../preprocessing/') #need this in order to get to the other file in other directory

#can comment out the ones you aren't using to save a little bit of time
from covidPreprocess import getCoronaText
from liarPreprocess import getLiarText
from fnnPreprocess import getFNNText

from preprocessingFunctions import getTermMatrixTestData

def saveModel(model, vectorizer):
    # Save to file in the current working directory
    pkl_filename = "pickle_model.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(model, file)
        pickle.dump(vectorizer, file)
    
def loadModel():
    # Load from file
    pkl_filename = "pickle_model.pkl"
    with open(pkl_filename, 'rb') as file:
        model = pickle.load(file)
        vec = pickle.load(file)
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
    


