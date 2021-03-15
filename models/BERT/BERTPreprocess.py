from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import re
import pickle

from transformers import AutoTokenizer

OPTIONS_NAME = "albert-base-v2"
FNN_FILENAME = os.path.join('..', '..', 'datasets', 'fake news detection(FakeNewsNet)', 'fnn_train.csv')
CORONA_FILENAME = os.path.join('..', '..', 'datasets', 'corona_fake.csv')
LIAR_FILENAME = os.path.join('..', '..', 'datasets', 'fake news detection(LIAR)', 'liar_test.csv')


def replaceCommas(strToRepl):
	'''
	function to search for numbers like 100,000 and replace them with 100000
	so that this number will stay combined when we perform tokenization
	'''
	if (not isinstance(strToRepl, str) and np.isnan([strToRepl])):
		return ""

    #searches to see if there is [digit,digit] in the text
	a = re.search(r'[0-9],[0-9]', strToRepl) 
	while (a != None): #if there is no more matches, then a will be None
		b = a.span()[0] + 1 #second character will be a comma (according to how we searched for it)
		strToRepl = strToRepl[:b] +  strToRepl[b+1:] #take everything but the comma
		a = re.search(r'[0-9],[0-9]', strToRepl)
	return strToRepl


def load_fnn_data():
	"""
	Tokenizes the FNN dataset using Albert tokkenizer
	"""
	data = pd.read_csv(FNN_FILENAME)

	texts = []
	labels = []

	# vectorize function to remove commas on array of texts
	commReplFunc = np.vectorize(replaceCommas)

	# makes the array of titles and article contents
	contents = commReplFunc(data['fullText_based_content'].to_numpy())
	titles = commReplFunc(data['statement'].to_numpy())
	texts = np.array([title + ' ' + content for title, content in zip(titles, contents)])

	# makes the array of labels (0 == false, 1 == true)
	labels = 1 * (data['label_fnn'].to_numpy() == "real")

	tokenizer = AutoTokenizer.from_pretrained(OPTIONS_NAME)
	train_tokens = tokenize_texts(tokenizer, texts)
	train_tokens_ids = np.array(list(map(tokenizer.convert_tokens_to_ids, train_tokens)))

	save_tokens_labels("fnn", train_tokens, train_tokens_ids, labels)

	return train_tokens, train_tokens_ids, labels


def load_corona_data():
	"""
	Tokenizes the Corona dataset using Albert tokkenizer
	"""
	data = pd.read_csv(CORONA_FILENAME)

	texts = []
	labels = []

	# vectorize function to remove commas on array of texts
	commReplFunc = np.vectorize(replaceCommas)

	# makes the array of titles and article contents
	contents = commReplFunc(data['text'].to_numpy())
	titles = commReplFunc(data['title'].to_numpy())
	texts = np.array([title + ' ' + content for title, content in zip(titles, contents)])

	# makes the array of labels (0 == false, 1 == true)
	labels = 1 * (data['label'].to_numpy() == "TRUE")

	tokenizer = AutoTokenizer.from_pretrained(OPTIONS_NAME)
	train_tokens = tokenize_texts(tokenizer, texts)
	train_tokens_ids = np.array(list(map(tokenizer.convert_tokens_to_ids, train_tokens)))

	save_tokens_labels("corona", train_tokens, train_tokens_ids, labels)

	return train_tokens, train_tokens_ids, labels
	

def tokenize_texts(tokenizer, texts):
	tokens = []
	for t in texts:
		# adds padding to the list to extend it to maximum length of 512
		token = tokenizer.tokenize(t, add_special_tokens=True, padding="max_length", truncation=True, max_length=512)

		# starts with [CLS] and ends with [SEP]
		# padding symbol is <sep> for Albert
		tokens.append(token)

	return np.array(tokens)

def save_tokens_labels(dir_name, tokens, token_ids, labels):
	if not os.path.exists(dir_name):
		os.makedirs(dir_name)

	pkl_filename = os.path.join(dir_name, "token.pkl")
	
	with open(pkl_filename, 'wb') as file:
		pickle.dump(tokens, file)
		pickle.dump(token_ids, file)
		pickle.dump(labels, file)


def load_tokens_labels(topic):
	assert topic in {'corona', 'fnn', 'liar'}
	assert os.path.exists(topic)
	filename = os.path.join(topic, 'token.pkl')

	with open(filename, 'rb') as file:
		tokens = pickle.load(file)
		token_ids = pickle.load(file)
		labels = pickle.load(file)

	return tokens, token_ids, labels

if __name__ == "__main__":
	load_fnn_data()
	load_corona_data()