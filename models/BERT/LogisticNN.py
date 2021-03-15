import torch
import numpy as np
import pickle
import torch
import torch.nn as nn

import os, sys

from BERTModel import load_albert_outputs
from BERTPreprocess import load_tokens_labels


class LogisticBinaryClassifier(nn.Module):
	def __init__(self, input_size, hidden_size1=400, hidden_size2=200):
		super(LogisticBinaryClassifier, self).__init__()
		self.fc1 = nn.Linear(input_size, hidden_size1)
		self.fc2 = nn.Linear(hidden_size1, hidden_size2)
		self.fc3 = nn.Linear(hidden_size2, 1)

	def forward(self, x):
		x = torch.tanh(self.fc1(x))
		x = torch.tanh(self.fc2(x))
		x = torch.sigmoid(self.fc3(x))
		return x
		

def train_model(topic, data=None, learning_rate=1e-02, epochs=500):
	"""
	Train the logistic neural network with given dataset or topic of the
	dataset
	"""
	assert topic in {'corona', 'fnn', 'liar'}

	if data is not None:
		token_ids, labels = data
	else:
		tokens, token_ids, labels = load_tokens_labels(topic)

	# albert_outputs = load_albert_outputs(topic)

	# print(token_ids[:5, :10])
	# print(albert_outputs[:5, :10])
	# return
	# print(token_ids, token_ids.shape, sep='\n')
	# print(labels, type(labels), labels.shape, sep='\n')
	# print(albert_outputs, type(albert_outputs), albert_outputs.shape, sep='\n')

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	model = LogisticBinaryClassifier(input_size=token_ids.shape[1]).to(device)
	optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
	loss_fn = nn.BCELoss()

	X_tensor = torch.from_numpy(token_ids).float()
	Y_tensor = torch.from_numpy(labels).float()

	train_data = torch.utils.data.TensorDataset(X_tensor, Y_tensor)
	train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

	model.train()

	for i in range(epochs):
		for step_num, (input_batch, label_batch) in enumerate(train_loader, start=1):
			input_batch = input_batch.to(device)
			label_batch = label_batch.to(device)
			output = model(input_batch)

			loss = loss_fn(output, label_batch.reshape(-1,1))

			# backward propagation
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		if i % 50 == 0:
			print("epoch {:<4} loss : {}".format(i, loss))
	
	model.eval()

	# X_tensor_samp = torch.from_numpy(albert_outputs[:50]).float()

	# predicted = model(X_tensor).cpu()
	# acc = (predicted.reshape(-1).detach().numpy().round() == labels).mean()

	# print('predicted:', predicted[:30])
	# print('labels:', labels[:30])
	# print('test result:', (predicted.reshape(-1).detach().numpy().round()[:30] == labels[:30]))
	# print('accruacy:', acc)

	# save_log_model(topic, model)

	return model


def save_log_model(topic, model):
	assert topic in {"corona", "fnn", "liar"}
	file_path = os.path.join(topic, "log_model.pkl")

	with open(file_path, 'wb') as file:
		pickle.dump(model, file)


if __name__ == '__main__':
	train_model('fnn')