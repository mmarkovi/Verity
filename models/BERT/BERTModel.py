from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd

# install package using "user> conda install -c huggingface transformers"
# more details at https://huggingface.co/transformers/
from transformers import BertTokenizer, BertForSequenceClassification, BertModel, AlbertModel, AlbertPreTrainedModel, AlbertConfig

from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

from torch.nn.utils import clip_grad_norm_
from IPython.display import clear_output
from sklearn.metrics import classification_report
import sys, os
import pickle

OPTIONS_NAME = "albert-base-v2"

from BERTPreprocess import load_tokens_labels

import time

# sys.path.insert(0, '../../preprocessing/') #need this in order to get to the other file in other directory
# from covidPreprocess import getCoronaVocabulary, getCoronaText

# using https://towardsdatascience.com/bert-to-the-rescue-17671379687f as reference

class BertBinaryClassifier(nn.Module):
    def __init__(self, dropout=0.1):
        super(BertBinaryClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path =
                                              OPTIONS_NAME)
        #self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, tokens):
        _, pooled_output = self.bert(tokens, output_hidden_states=False)
        print(pooled_output)
        #_, pooled_output = self.bert(tokens, attention_mask=masks, output_all_encoded_layers=False)
        # have to add "masks = None" to the parameters to use the above line
        #dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(pooled_output)
        proba = self.sigmoid(linear_output)
        return proba

#made using help from
# towardsdatascience.com/fine-tune-albert-with-pre-training-on-custom-corpus-f56ea3cfdc82 using below github
# https://github.com/LydiaXiaohongLi/Albert_Finetune_with_Pretrain_on_Custom_Corpus/blob/master/Albert_Finetune_with_Pretrain_on_Custom_Corpus_ToyModel.ipynb


class ALBERTModel(AlbertPreTrainedModel):
	def __init__(self, config):
		super().__init__(config = config)

		# using small albert set
		options_name = "albert-base-v2"
		self.encoder = AlbertModel.from_pretrained(options_name)
		self.predictions_dense = nn.Linear(config.hidden_size, 768)
		self.predictions_LayerNorm = nn.LayerNorm(768)
		self.predictions_bias = nn.Parameter(torch.zeros(5120)) 
		self.predictions_decoder = nn.Linear(768, 5120)
		self.sigmoid = nn.Sigmoid()

	def forward(self, text):
		loss, text_fea = self.encoder(text)[:2]
		loss = self.predictions_dense(loss)
		loss = self.predictions_LayerNorm(loss)
		prediction_scores = self.predictions_decoder(loss)
		#print("pred:", prediction_scores)
		return prediction_scores#, text_fea
    
    
def load_and_process_data():
    X,Y = getCoronaText()
    X_train, Y_train = getCoronaText(True)
    
    #using a subset because it takes a long time to run
    X = X[:5]
    Y = Y[:5]
    X_train = X_train[:5]
    Y_train = Y_train[:5]
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    
    #cutting/padding out the values to be 512 words, since that is the max amount BERT allows
    train_tokens = list(map(lambda t: ['[CLS]'] + tokenizer.tokenize(t)[:510] + ['[SEP]']
                            + [0] * (510 - len(tokenizer.tokenize(t)[:510])), X_train))
    test_tokens = list(map(lambda t: ['[CLS]'] + tokenizer.tokenize(t)[:510] + ['[SEP]']
                           + [0] * (510 - len(tokenizer.tokenize(t)[:510])), X))

    
    train_tokens_ids = list(map(tokenizer.convert_tokens_to_ids, train_tokens))
    test_tokens_ids = list(map(tokenizer.convert_tokens_to_ids, test_tokens))

    X_tensor = torch.tensor(train_tokens_ids) #torch.from_numpy(X_train.todense()).float().to(torch.int64)
    Y_tensor = torch.from_numpy(np.array(Y_train)).to(torch.int64)
    
    X_test_tensor = torch.tensor(test_tokens_ids) #torch.from_numpy(x_test.todense()).float().to(torch.int64)
    Y_test_tensor = torch.from_numpy(np.array(Y)).to(torch.int64)
    
    #could probably use cuda, but was running into memory issues
    
    device = torch.device('cpu')
    #use TensorDataset to be able to use our DataLoader
    train_data = torch.utils.data.TensorDataset(X_tensor, Y_tensor)
    train_loader = torch.utils.data.DataLoader(train_data,batch_size=16, shuffle=True)
    
    test_data = torch.utils.data.TensorDataset(X_test_tensor, Y_test_tensor)
    test_loader = torch.utils.data.DataLoader(test_data,batch_size=16, shuffle=True)
    
    #vocabsize = X_train.shape[1]
    num_epochs = 1
    config = AlbertConfig.from_pretrained("albert-base-v2")
    bert_clf = ALBERTModel(config)
    #bert_clf = bert_clf.cuda()
    optimizer = torch.optim.Adam(bert_clf.parameters(), lr=1e-06)
    BATCH_SIZE = 16
    
    for epoch_num in range(num_epochs):
        bert_clf.train()
        train_loss = 0
        for step_num, batch_data in enumerate(train_loader):
            token_ids, labels = tuple(t.to(device) for t in batch_data)
            #print(str(torch.cuda.memory_allocated(device)/1000000 ) + 'M')
            #print(token_ids)
            #print(labels)
            logits = bert_clf(token_ids)
            
            loss_func = nn.CrossEntropyLoss()
    
            batch_loss = loss_func(logits.view(labels.shape[0], -1), labels)
            train_loss += batch_loss.item()
            
            
            bert_clf.zero_grad()
            batch_loss.backward()
            
    
            clip_grad_norm_(parameters=bert_clf.parameters(), max_norm=1.0)
            optimizer.step()
            
            clear_output(wait=True)
            print('Epoch: ', epoch_num + 1)
            print("\r" + "{0}/{1} loss: {2} ".format(step_num, len(train_data) / BATCH_SIZE, train_loss / (step_num + 1)))

    bert_clf.eval()
    bert_predicted = []
    all_logits = []
    with torch.no_grad():
        for step_num, batch_data in enumerate(test_loader):
    
            token_ids, labels = tuple(t.to(device) for t in batch_data)
    
            logits = bert_clf(token_ids)
            loss_func = nn.CrossEntropyLoss()
            loss = loss_func(logits.view(labels.shape[0], -1), labels)
            numpy_logits = logits.view(labels.shape[0], -1).cpu().detach().numpy()
            
            print(numpy_logits)
            tp = np.zeros(numpy_logits[:, 0].shape)
            tp[numpy_logits[:, 0] > 0.5] = 1
            bert_predicted += list(tp)
            all_logits += list(numpy_logits[:, 0])
    Y_test = np.array(Y, dtype = float)
    print(np.mean(bert_predicted == Y_test))
    print(classification_report(Y_test, bert_predicted))

def train_corona_model():
	corona_token_filename = os.path.join('corona', 'token.pkl')

	tokens, token_ids, labels = load_tokens_labels(corona_token_filename)

	# print('tokens:', tokens[:1], len(tokens), sep='\n')
	# print('token_ids:', token_ids[:1], len(token_ids), sep='\n')
	# print('labels:', labels[:10], len(labels), sep='\n')

	albert = AlbertModel.from_pretrained(OPTIONS_NAME)

	X_tensor = torch.from_numpy(token_ids).to(torch.int64)
	Y_tensor = torch.from_numpy(labels).to(torch.int64)

	train_data = torch.utils.data.TensorDataset(X_tensor, Y_tensor)
	train_loader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True)

	for step_num, (ids, labels) in enumerate(train_loader, start=1):
		start = time.time()
		outputs = albert(ids)
		end = time.time()

		print('it took', end - start, 'seconds')
		print(outputs)
		break
	
	# print('x shape:', x.shape)
	# print('y:', y)
	# print('y shape:', y.shape)
	# print('pooled:', pooled)
	# print('pooled shape:', pooled.shape)




def save_model(dir_name, model):
	version_num = 1
	while True:
		file_path = os.path.join(dir_name, "model" + str(version_num) + ".pkl")
		if os.path.exists(file_path):
			version_num += 1
		else:
			break
	
	with open(file_path, 'wb') as file:
		pickle.dump(model, file)
    
if __name__ == "__main__":
    train_corona_model()