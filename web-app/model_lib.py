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

	def forward(self, x):
		out = self.hidden1(x)
		out = self.relu(out)
		out = self.hOutput1(out)
		return out

def load_model():
    # Load from file
    pkl_filename = "model/pickle_vec.pkl"
    model_filename = "model/saved_model"
    with open(pkl_filename, 'rb') as file:
        vec = pickle.load(file)

    model = torch.load(model_filename)
    return model, vec

def predict_model(model, vec, raw_text):
    # transform raw text with the vectorizer
	text = pf.getTermMatrixTestData(raw_text, vec).todense()

	vocabsize = text.shape[1]

	dummy = np.repeat(1, vocabsize * 15)
	print(text.A1.shape)
	dummy_text_matrix = np.concatenate([text.A1, dummy]).reshape(16, vocabsize)

	print(dummy_text_matrix)
	print(dummy_text_matrix.reshape(16, vocabsize).shape)
    
	X_test_tensor = torch.from_numpy(text).float()

	print(X_test_tensor)
	print(X_test_tensor.shape)

	#test against model
	output = model(X_test_tensor)
	_, predicted = torch.max(output.data, 1)
	Ytest = predicted.numpy()[0]



	prob_false = output.data.numpy()[0][0] / sum(output.data.numpy()[0, :])

	print(output)

	print(predicted)
	return Ytest, prob_false

if __name__ == "__main__":
    model, vec = load_model()
    predict_model(model, vec, "You just need to add water, and the drugs and vaccines are ready to be administered. There are two parts to the kit: one holds pellets containing the chemical machinery that synthesises the end product, and the other holds pellets containing instructions that telll the drug which compound to create. Mix two parts together in a chosen combination, add water, and the treatment is ready.")