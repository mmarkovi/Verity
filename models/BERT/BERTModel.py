from __future__ import unicode_literals, print_function, division
import torch
from torch._C import device
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd

# install package using "user> conda install -c huggingface transformers"
# more details at https://huggingface.co/transformers/
from transformers import AlbertForSequenceClassification, AlbertModel, AlbertPreTrainedModel, AlbertConfig
from transformers import AdamW, get_linear_schedule_with_warmup


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


# using https://medium.com/@aniruddha.choudhury94/part-2-bert-fine-tuning-tutorial-with-pytorch-for-text-classification-on-the-corpus-of-linguistic-18057ce330e1 as reference
def fine_tune_albert(topic="corona"):
	assert topic in {"corona", "fnn", "liar"}

	tokens, token_ids, labels = load_tokens_labels(topic)

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	# device = "cpu"

	model = AlbertForSequenceClassification.from_pretrained(OPTIONS_NAME, num_labels=1, output_attentions=False, output_hidden_states=False).to(device)

	X_tensor = torch.from_numpy(token_ids).long()
	Y_tensor = torch.from_numpy(labels).float()

	train_data = torch.utils.data.TensorDataset(X_tensor, Y_tensor)
	train_loader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True)

	b_size = 16
	learning_rate = 5e-5
	num_epoch = 3

	optimizer = AdamW(model.parameters(), lr = learning_rate, eps = 1e-8)
	total_steps = len(train_loader) * num_epoch
	scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps = total_steps)

	loss_values = []

	for epoch_i in range(num_epoch):
		print("")
		print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, num_epoch))
		print('Training...')

		# Measure how long the training epoch takes.
		t0 = time.time()
		# Reset the total loss for this epoch.
		total_loss = 0
		# Put the model into training mode.
		model.train()

		for step, batch in enumerate(train_loader):
			# Progress update every 40 batches.
			if step % 40 == 0 and not step == 0:
				# Calculate elapsed time in minutes.
				elapsed = time.time() - t0
				
				# Report progress.
				print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:.2f} seconds.'.format(step, len(train_loader), elapsed))

			b_input_ids, b_labels = batch
			b_input_ids = b_input_ids.to(device)
			b_labels = b_labels.to(device)

			# print(torch.cuda.memory_summary(device=device, abbreviated=False))
			
			model.zero_grad()        
			# Perform a forward pass (evaluate the model on this training batch).
			# This will return the loss (rather than the model output) because we
			# have provided the `labels`.
			# The documentation for this `model` function is here: 
			# https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
			outputs = model(b_input_ids, token_type_ids=None, labels=b_labels)
			
			# The call to `model` always returns a tuple, so we need to pull the 
			# loss value out of the tuple.
			loss = outputs[0]
			# Accumulate the training loss over all of the batches so that we can
			# calculate the average loss at the end. `loss` is a Tensor containing a
			# single value; the `.item()` function just returns the Python value 
			# from the tensor.
			total_loss += loss.item()
			# Perform a backward pass to calculate the gradients.
			loss.backward()
			# Clip the norm of the gradients to 1.0.
			# This is to help prevent the "exploding gradients" problem.
			torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
			# Update parameters and take a step using the computed gradient.
			# The optimizer dictates the "update rule"--how the parameters are
			# modified based on their gradients, the learning rate, etc.
			optimizer.step()
			# Update the learning rate.
			scheduler.step()
		# Calculate the average loss over the training data.
		avg_train_loss = total_loss / len(train_loader)            
		
		# Store the loss value for plotting the learning curve.
		loss_values.append(avg_train_loss)
		print("")
		print("  Average training loss: {0:.2f}".format(avg_train_loss))
		print("  Training epoch took: {:<.2f} seconds".format(time.time() - t0))
			
		
	print("")
	print("Training complete!")


	save_albert_model(topic, model)

	return model





def get_albert_outputs(topic="corona"):
	assert topic in {"corona", "fnn", "liar"}

	tokens, token_ids, labels = load_tokens_labels(topic)

	# print('tokens:', tokens[:1], len(tokens), sep='\n')
	# print('token_ids:', token_ids[:1], len(token_ids), sep='\n')
	# print('labels:', labels[:10], len(labels), sep='\n')
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	albert = AlbertForSequenceClassification.from_pretrained(OPTIONS_NAME, num_labels=1, output_attentions=False, output_hidden_states=False).to(device)

	X_tensor = torch.from_numpy(token_ids).to(torch.int64)
	Y_tensor = torch.from_numpy(labels).to(torch.int64)

	train_data = torch.utils.data.TensorDataset(X_tensor, Y_tensor)
	train_loader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True)

	aggregated_outputs = None

	total_iterations = len(labels) // 16 + 1

	start = time.time()
	for step_num, (ids, labels) in enumerate(train_loader, start=1):
		outputs = albert(ids)

		pooled = outputs['pooler_output'].detach().numpy()

		if step_num > 1:
			aggregated_outputs = np.concatenate((aggregated_outputs, pooled))
		else:
			aggregated_outputs = pooled
		
		print('Finished Step {} out of {}'.format(step_num, total_iterations))

	save_albert_outputs(topic, aggregated_outputs)
	
	end = time.time()
	print('It took {:.2f} seconds'.format(end - start))

	return aggregated_outputs

"""
Helper Functions
"""

def find_accuracy(preds, labels):
	comparisons = preds.reshape(-1).detach().numpy().round() == labels
	return comparisons.mean()


"""
Pickle Methods (Save & Load)
"""

def save_albert_outputs(dir_name, out):
	file_path = os.path.join(dir_name, "albert_out.pkl")

	with open(file_path, 'wb') as file:
		pickle.dump(out, file)


def load_albert_outputs(topic='corona'):
	assert topic in {"corona", "fnn", "liar"}
	file_path = os.path.join(topic, 'albert_out.pkl')
	
	with open(file_path, 'rb') as file:
		out = pickle.load(file)

	return out


def save_albert_model(dir_name, model):
	file_path = os.path.join(dir_name, "albert_model.pkl")

	with open(file_path, 'wb') as file:
		pickle.dump(model, file)


def load_albert_model(topic='corona'):
	assert topic in {"corona", "fnn", "liar"}
	file_path = os.path.join(topic, 'albert_model.pkl')
	
	with open(file_path, 'rb') as file:
		out = pickle.load(file)

	return out

if __name__ == "__main__":
    fine_tune_albert(topic="corona")