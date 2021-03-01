from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
from torchtext.data import Field, TabularDataset, BucketIterator, Iterator

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import re
import pickle

from transformers import AutoTokenizer

OPTIONS_NAME = "albert-base-v2"
FNN_FILENAME = os.path.join('..', '..', 'datasets', 'fake news detection(FakeNewsNet)', 'fnn_train.csv')


def replaceCommas(strToRepl):
    '''
	function to search for numbers like 100,000 and replace them with 100000
    so that this number will stay combined when we perform tokenization
    '''
    #searches to see if there is [digit,digit] in the text
    a = re.search(r'[0-9],[0-9]', strToRepl) 
    while (a != None): #if there is no more matches, then a will be None
        b = a.span()[0] + 1 #second character will be a comma (according to how we searched for it)
        strToRepl = strToRepl[:b] +  strToRepl[b+1:] #take everything but the comma
        a = re.search(r'[0-9],[0-9]', strToRepl)
    return strToRepl


def load_data():
	data = pd.read_csv(FNN_FILENAME)

	texts = []
	labels = []

	# vectorize function to remove commas on array of texts
	commReplFunc = np.vectorize(replaceCommas)

	# makes the array of titles and article contents
	contents = commReplFunc(data['fullText_based_content'].to_numpy())
	titles = data['statement'].to_numpy()
	texts = np.array([title + ' ' + content for title, content in zip(titles, contents)])

	# makes the array of labels (0 == false, 1 == true)
	labels = data['label_fnn'].to_numpy == "real"

	# print('TEXTS:')
	# print(texts[:2], texts.shape,sep='\n')
	# print('LABELS:')
	# print(labels[:2], labels.shape,sep='\n')

	tokenizer = AutoTokenizer.from_pretrained(OPTIONS_NAME)
	train_tokens = list(map(lambda t: ['[CLS]'] + tokenizer.tokenize(t)[:511], texts))
	train_tokens_ids = list(map(tokenizer.convert_tokens_to_ids, train_tokens))

	save_tokens_labels(train_tokens, train_tokens_ids, labels)

	return train_tokens, train_tokens_ids, labels
	

def save_tokens_labels(tokens, token_ids, labels):
	pkl_filename = "token.pkl"
	
	with open(pkl_filename, 'wb') as file:
		pickle.dump(tokens, file)
		pickle.dump(token_ids, file)
		pickle.dump(labels, file)

def load_tokens_labels(filename):
	with open(filename, 'rb') as file:
		tokens = pickle.load(file)
		token_ids = pickle.load(file)
		labels = pickle.load(file)

	return tokens, token_ids, labels

if __name__ == "__main__":
	load_data()