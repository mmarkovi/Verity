# TODO: make a function for loading up the saved vectorizer and model using pickle
# TODO: make a function for transforming a raw text with the vectorizer and pickle
# TODO: make a function for predicting the truthfulness of the article (with percentage of truthfulness)

import pickle, os
import numpy as np
import torch
import torch.nn as nn
import preprocessingFunctions as pf

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

def load_model():
    # Load from file
    pkl_filename = os.path.join("model", "pickle_model.pkl")
    with open(pkl_filename, 'rb') as file:
        model = pickle.load(file)
        vec = pickle.load(file)

        print(type(model))
        print(type(vec))
    return model, vec

def predict_model(model, vec, raw_text):
    # transform raw text into the vectorizer
    text = pf.getTermMatrixTestData(raw_text, vec)
    X_test_tensor = torch.from_numpy(text.todense()).float()

    #test against model
    output = model(X_test_tensor)
    _, predicted = torch.max(output.data, 1)
    Ytest = predicted.numpy()[0]

    print(output)

    print(Ytest)
    return Ytest

if __name__ == "__main__":
    model, vec = load_model()
    predict_model(model, vec, "China virus affected US terrorism")