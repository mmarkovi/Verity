from __future__ import unicode_literals, print_function, division
from models.BERT.BERTPreprocess import OPTIONS_NAME
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd

# install package using "user> conda install -c huggingface transformers"
# more details at https://huggingface.co/transformers/
from transformers import BertTokenizer, BertForSequenceClassification

from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

import sys

OPTIONS_NAME = "albert-base-v2"

# using https://towardsdatascience.com/bert-to-the-rescue-17671379687f as reference

class BertBinaryClassifier(nn.Module):
    def __init__(self, dropout=0.1):
        super(BertBinaryClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.linear = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, tokens):
        _, pooled_output = self.bert(tokens, utput_all=False)
        linear_output = self.linear(dropout_output)
        proba = self.sigmoid(linear_output)
        return proba

class ALBERTModel(nn.Module):
	def __init__(self):
		super(BERTModel, self).__init__()

		# using small albert set
		options_name = "albert-base-v1"
		self.encoder = BertForSequenceClassification.from_pretrained(options_name)

	def forward(self, text, label):
		loss, text_fea = self.encoder(text, labels=label)[:2]

		return loss, text_fea

def load_and_process_data():
	X,Y = getFNNText()
	X_train, Y_train, vectorizer_train = getFNNVocabulary(True)
