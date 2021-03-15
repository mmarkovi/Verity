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
from nltk.corpus import stopwords
#nltk.download('stopwords')

from covidPreprocess import getCoronaVocabulary, getCoronaText, get_whole_Corona_dataset
from liarPreprocess import getLiarVocabulary, getLiarText
from fnnPreprocess import getFNNVocabulary, getFNNText


def getAllVocabulary(isTrain = False):
    '''
    Gets and preprocesses data for topic classification of raw text between covid-related and not covid-related.
    Combines data from Corona, Liar, and FNN datasets. All data from Corona dataset set to have label 1. 
    All data from Liar and FNN datasets set to have label 0.
    
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

    text, Y, vectorizer = getAllText(isTrain)
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

def getAllText(isTrain = False):
    '''
    Gets and preprocesses data for topic classification of raw text between covid-related and not covid-related.
    Combines data from Corona, Liar, and FNN datasets. All data from Corona dataset set to have label 1. 
    All data from Liar and FNN datasets set to have label 0.
    
    Preprocesses data in a manner suitable for Pytorch neural networks.
    '''
    text_corona, Y_corona = getCoronaText(isTrain)
    text_liar, Y_liar = getLiarText(isTrain)
    text_fnn, Y_fnn = getFNNText(isTrain)

    text_corona = np.array(text_corona)
    Y_corona = np.array(Y_corona)
    Y_corona = np.ones(Y_corona.size, dtype=int)

    text_liar = np.array(text_liar)
    Y_liar = np.array(Y_liar)
    Y_liar = np.zeros(Y_liar.size, dtype=int)

    text_fnn = np.array(text_fnn)
    Y_fnn = np.array(Y_fnn)
    Y_fnn = np.zeros(Y_fnn.size, dtype=int)

    text_all = np.append(text_corona, text_liar)
    text_all = np.append(text_all, text_fnn)
    Y_all = np.append(Y_corona, Y_liar)
    Y_all = np.append(Y_all, Y_fnn)

    text_all = list(text_all)
    Y_all = list(Y_all)

    # vectorizerText = CountVectorizer(stop_words = pf.getLemmatizedStopwords(), min_df=.01, ngram_range=(1,2), tokenizer= pf.LemmaTokenizer())
    # X = vectorizerText.fit_transform(text_all)
    vectorizerText = None

    return text_all, Y_all, vectorizerText

    # return X, Y_all, vectorizerText

def getAllText2(isTrain = False):
    '''
    Gets and preprocesses data for topic classification of raw text between covid-related and not covid-related.
    Combines data from Corona, Liar, and FNN datasets. All data from Corona dataset set to have label 1. 
    All data from Liar and FNN datasets set to have label 0.
    
    Preprocesses data in a manner suitable for sklearn logistic regression.
    '''
    text_corona, Y_corona = getCoronaText(isTrain)
    text_liar, Y_liar = getLiarText(isTrain)
    text_fnn, Y_fnn = getFNNText(isTrain)

    text_corona = np.array(text_corona)
    Y_corona = np.array(Y_corona)
    Y_corona = np.ones(Y_corona.size, dtype=int)

    text_liar = np.array(text_liar)
    Y_liar = np.array(Y_liar)
    Y_liar = np.zeros(Y_liar.size, dtype=int)

    text_fnn = np.array(text_fnn)
    Y_fnn = np.array(Y_fnn)
    Y_fnn = np.zeros(Y_fnn.size, dtype=int)

    text_all = np.append(text_corona, text_liar)
    text_all = np.append(text_all, text_fnn)
    Y_all = np.append(Y_corona, Y_liar)
    Y_all = np.append(Y_all, Y_fnn)

    vectorizerText = CountVectorizer(stop_words = pf.getLemmatizedStopwords(), min_df=.01, ngram_range=(1,2), tokenizer= pf.LemmaTokenizer())
    X = vectorizerText.fit_transform(text_all)

    return X, Y_all, vectorizerText