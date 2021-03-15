import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split, StratifiedKFold

import os, sys

from BERTModel import load_albert_outputs
from BERTPreprocess import load_tokens_labels
from LogisticNN import LogisticBinaryClassifier, train_model


def validate_LogisticNN(topic, debug=False):
	assert topic in {'corona', 'fnn', 'liar'}
	tokens, token_ids, labels = load_tokens_labels(topic)

	epoch_list = [50, 100, 250]
	numFold = 3

	total_train_acc_list = []
	total_test_acc_list = []

	for n_epoch in epoch_list:
		print('beginning a test for', n_epoch, "epoches")
		skf = StratifiedKFold(n_splits=numFold)
		skf.get_n_splits(token_ids, labels)

		total_train_acc = 0
		total_test_acc = 0
		
		for i, (train_ind, test_ind) in enumerate(skf.split(token_ids, labels), start=1):
			X_train, X_test = token_ids[train_ind], token_ids[test_ind]
			Y_train, Y_test = labels[train_ind], labels[test_ind]

			# print('X train shape:', X_train.shape)
			# print('X test shape:', X_test.shape)
			# print('Y train shape:', Y_train.shape)
			# print('Y test shape:', Y_test.shape)
			
			model = train_model(topic, data=(X_train, Y_train), epochs=n_epoch)

			X_train_tensor = torch.from_numpy(X_train).float()
			Y_train_tensor = torch.from_numpy(Y_train).float()
			X_test_tensor = torch.from_numpy(X_test).float()
			Y_test_tensor = torch.from_numpy(Y_test).float()

            # use TensorDataset to be able to use our DataLoader
			train_data = torch.utils.data.TensorDataset(X_train_tensor, Y_train_tensor)
			train_loader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=False)
			test_data = torch.utils.data.TensorDataset(X_test_tensor, Y_test_tensor)
			test_loader = torch.utils.data.DataLoader(test_data, batch_size=16, shuffle=False)

			train_acc, test_acc = get_model_accuracy(train_loader, test_loader, model)
			total_train_acc += train_acc
			total_test_acc += test_acc

			# print(i, '-Fold:', sep='')
			# print("train accuracy: {:.4f}%".format(train_acc * 100))
			# print("test accuracy: {:.4f}%".format(test_acc * 100))
			# print("difference in accuracies: {:.4f}%".format(abs(test_acc - train_acc) * 100))

		total_train_acc /= numFold
		total_test_acc /= numFold

		total_train_acc_list.append(total_train_acc)
		total_test_acc_list.append(total_test_acc)

		if debug:
			print("final train accuracy: {:.4f}%".format(total_train_acc * 100))
			print("final test accuracy: {:.4f}%".format(total_test_acc * 100))

	if debug:
		print("tr:", total_train_acc_list)
		print("te:", total_test_acc_list)

	total_train_acc_list = np.array(total_train_acc_list) * 100
	total_test_acc_list = np.array(total_test_acc_list) * 100

	plt_data = pd.DataFrame({'Number of Epoch': epoch_list, 'Train Accuracy': total_train_acc_list, \
							'Test Accuracy': total_test_acc_list})
	plt.plot('Number of Epoch', 'Train Accuracy', data=plt_data, marker='.', markerfacecolor='skyblue', markersize=10, color='skyblue', linewidth=2)
	plt.plot('Number of Epoch', 'Test Accuracy', data=plt_data, marker='.', markerfacecolor='olive', markersize=10, color='olive', linewidth=2)
	plt.xlabel('Number of Epoch')
	plt.ylabel('Accuracy')
	plt.legend()
	plt.show()

	return model


def get_model_accuracy(train_loader, test_loader, model, debug=False):
    # Test the model
    # In the test phase, we don't need to compute gradients (the model has already been learned)
	train_accuracy = 0
	test_accuracy = 0
	
	model.cpu()

	with torch.no_grad():
		total  = 0
		correct = 0

		for data, labels in train_loader:
			predicted = model(data)
			labels = labels.detach().numpy()
			predicted_numpy = predicted.reshape(-1).detach().numpy().round()
			# print(predicted_numpy, predicted_numpy.shape, type(predicted_numpy))
			# print(labels, labels.shape, type(labels))
			correct += (predicted_numpy == labels).sum().item()
			total += labels.shape[0]

			if debug:
				print('predicted:', predicted)


		train_accuracy = correct / total

		total = 0
		correct = 0

		for data, labels in test_loader:
			predicted = model(data)
			labels = labels.detach().numpy()
			predicted_numpy = predicted.reshape(-1).detach().numpy().round()
			correct += (predicted_numpy == labels).sum().item()
			total += labels.shape[0]

		test_accuracy = correct / total

		if debug:
			print("train accuracy: {:.4f}%".format(train_accuracy * 100))
			print("test accuracy: {:.4f}%".format(test_accuracy * 100))
			print("difference in accuracies: {:.4f}%".format(abs(test_accuracy - train_accuracy) * 100))

	return train_accuracy, test_accuracy


if __name__ == "__main__":
	validate_LogisticNN('fnn')