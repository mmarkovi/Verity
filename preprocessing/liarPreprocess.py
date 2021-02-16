#Written by Mia Markovic (mmarkovi, 39425669) using base code provided for assignment 1 for CS 175
 
# ---------------------------------------------------------------------------------------
# CS 175, WINTER 2021: VERITY LIAR DATA PREPROCESSING
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

liarfile =  pd.read_csv("../datasets/fake news detection(LIAR)/liar_test.csv")

liarfileTrain =  pd.read_csv("../datasets/fake news detection(LIAR)/liar_train.csv")



def getLiarVocabulary(isTrain = False):
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
    
    text, Y = getLiarText(isTrain)
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

def getLiarText(isTrain = False):
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
    lfile = liarfileTrain if isTrain else liarfile
    breakI = 15052 if isTrain else 1266
    print('\nExtracting tokens from each review.....(can be slow for a large number of reviews)......')   
    for d in lfile.loc:
        ftext = d['fullText_based_content']     # keep only the text and label
        ftitle = d['statement']
        label = (d['label-liar']).lower()
        
        #possible labels for LIAR:
        # {'pants-fire', 'barely-true', 'true', 'half-true', 'false', 'mostly-true'}
        score = 0 #1 for true, 0 for fake
        if (label == "true" or label == "mostly-true"): #rest of labels will be considered false
            score = 1
            
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
        if (i == breakI): #1266 for test, 15052 for train
            #for some reason the for loop doesnt know when to stop so put in a manual break
            break
    #print("there are", nanTitle, "nan titles")
    #print("there are", nanText, "nan text")
    return text, Y

if __name__ == "__main__":
    getLiarVocabulary()