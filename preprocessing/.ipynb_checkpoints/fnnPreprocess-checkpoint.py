#Written by Mia Markovic (mmarkovi, 39425669) using base code provided for assignment 1 for CS 175
 
# ---------------------------------------------------------------------------------------
# CS 175, WINTER 2021: VERITY FAKENEWSNET DATA PREPROCESSING
#
# ---------------------------------------------------------------------------------------

import nltk 
from nltk import word_tokenize
import simplejson as json
import sklearn
from sklearn.feature_extraction.text import * 
from sklearn.model_selection import train_test_split 

from sklearn import linear_model 
from sklearn import metrics 

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import preprocessingFunctions as pf

fnnfile =  pd.read_csv("../datasets/fake news detection(FakeNewsNet)/fnn_test.csv")

fnnfileTrain =  pd.read_csv("../datasets/fake news detection(FakeNewsNet)/fnn_train.csv")



def getFNNVocabulary(isTrain = False):
    '''

    Parameters
    ----------
    isTrain : bool, optional
        Boolean to tell the program if we want to look at the training dataset (true)
        or the testing dataset (false). The default is False.

    Returns
    -------
    X : NxM Array
        Returns a NxM matrix, where N = number of documents, M = size of vocabulary.
        The array contains the documetn term matrix for our current dataset.
    Y : TYPE
        A list of integers (0 or 1) describing which class a certain document is from.
        0 = fake article, 1 = true article
    vectorizer : CountVectorizer
        The BOW for our current dataset.
        
    '''
    
    text, Y = getFNNText(isTrain)    
    # create an instance of a CountVectorizer, using 
    # (1) the standard 'english' stopword set from nltk, but lemmetized
    # (2) only keeping terms in the vocabulary that occur in at least 1% of documents
    # (3) allowing both unigrams and bigrams in the vocabulary (use "ngram_range=(1,2)" to do this)
    vectorizerText = CountVectorizer(stop_words = pf.getLemmatizedStopwords(), min_df=.01, ngram_range=(1,2), tokenizer= pf.LemmaTokenizer() )
    # create a sparse BOW array from 'text' using vectorizer  
    
    X = vectorizerText.fit_transform(text)
    
    print('Data shape for text: ', X.shape)
    
    #can comment out to not see the vocabularies
    #print('Vocabulary for text: ', vectorizerText.get_feature_names())

    return X, Y, vectorizerText

def getFNNText(isTrain = False):
    '''

    Parameters
    ----------
    isTrain : bool, optional
        Boolean to tell the program if we want to look at the training dataset (true)
        or the testing dataset (false). The default is False.

    Returns
    -------
    text : TYPE
        A list of the imporant text (title, text) from the corresponding dataset
    Y : TYPE
        A list of integers (0 or 1) describing which class a certain document is from.
        0 = fake article, 1 = true article

    '''
    text = []
    Y = []
    i = 0
    nanTitle = 0
    nanText = 0
    ffile = fnnfileTrain if isTrain else fnnfile
    breakI = 15212 if isTrain else 1054
    print('\nExtracting tokens....')     
    for d in ffile.loc:
        ftext = d['fullText_based_content']     # keep only the text and label
        ftitle = d['statement']
        label = (d['label_fnn']).lower()
        
        score = 1 #1 for true, 0 for fake
        if (label == "fake"):
            score = 0
            
        #some documents might not have titles (or possible text?)
        #these are stored as NaN so replace with an empty string
        if (not isinstance(ftext, str) and np.isnan([ftext])):
            ftext = ""
            nanText += 1
        if (not isinstance(ftitle, str) and np.isnan(ftitle)):
            ftitle = ""
            nanTitle += 1
        
        ftext = ftext + ftitle #combining the text and title into one
        ftext = pf.replaceCommas(ftext)
            
        text.append(ftext)
        Y.append(score)
        i += 1
        if (i == breakI):   #1054 for test data, 15212 for training
            # for some reason the for loop doesnt know when to stop so put in a manual break
            break
    #print("there are", nanTitle, "nan titles")
    #print("there are", nanText, "nan text")
    return text, Y

if __name__ == "__main__":
    getFNNVocabulary()