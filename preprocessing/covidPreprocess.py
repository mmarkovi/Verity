#Written by Mia Markovic (mmarkovi, 39425669) using base code provided for assignment 1 for CS 175
 
# ---------------------------------------------------------------------------------------
# CS 175, WINTER 2021: VERITY COVID DATA PREPROCESSING
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
from nltk.corpus import stopwords
#nltk.download('stopwords')

coronafile =  pd.read_csv("../../datasets/corona_fake.csv")


#cleaning up broken data according to labels added by
# https://towardsdatascience.com/explore-covid-19-infodemic-2d1ceaae2306
coronafile.loc[5]['label'] = 'fake'
coronafile.loc[15]['label'] = 'true'
coronafile.loc[43]['label'] = 'fake'
coronafile.loc[131]['label'] = 'true'
coronafile.loc[242]['label'] = 'fake'

#splitting up into training and testing
#sample with fraction of 1 will shuffle the dataset
# the indexes will stay from the original dataset so calling reset_index will renumber them
coronafileTrain = coronafile.sample(frac = 1, random_state=1).reset_index(drop = True) 

originalSize = coronafile.shape[0]
splitSize = int(originalSize * .75) #873 of the 1164 documents will go to training, rest test
#reset index will make indices go from 0 to n, not be the shuffled original indices



def getCoronaVocabulary(isTrain = False, debug=False):
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
    
    text, Y = getCoronaText(isTrain)    
    # create an instance of a CountVectorizer, using 
    # (1) the standard 'english' stopword set from nltk, but lemmetized
    # (2) only keeping terms in the vocabulary that occur in at least 1% of documents
    # (3) allowing both unigrams and bigrams in the vocabulary (use "ngram_range=(1,2)" to do this)
    vectorizerText = CountVectorizer(stop_words = pf.getLemmatizedStopwords(), min_df=.01, ngram_range=(1,2), tokenizer= pf.LemmaTokenizer() )
    # create a sparse BOW array from 'text' using vectorizer  
    X = vectorizerText.fit_transform(text)
    
    if debug:
        print('Data shape for text: ', X.shape)
    #print('Vocabulary for text: ', vectorizerText.get_feature_names())

    return X, Y, vectorizerText


def getCoronaText(isTrain = False, debug=False):
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
    coronafile_train = coronafile.sample(frac = 1, random_state=1).reset_index(drop = True)

    originalSize = coronafile.shape[0]
    splitSize = int(originalSize * .75) #873 of the 1164 documents will go to training, rest test

    coronafile_test = coronafile_train.loc[:splitSize-1,:] #goes inclusive to the last one, so subtract 1
    coronafile_train = coronafile_train.loc[splitSize:,:].reset_index(drop = True) 

    text = []
    Y = []
    i = 0
    nanTitle = 0
    nanText = 0
    cFile = coronafile_test
    breakI = splitSize
    if (isTrain):
        cFile = coronafile_train
        breakI = originalSize - splitSize

    if debug:
        print('\nExtracting tokens....')   

    for index, d in cFile.iterrows():
        ftext = d['text']   # keep only the text and label
        ftitle = d['title']
        label = (d['label']).lower()
        
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
        if (i == breakI):
            #for some reason the for loop doesnt know when to stop so put in a manual break
            break
    if debug:
        print("there are", nanTitle, "nan titles")
        print("there are", nanText, "nan text")
    return text, Y

def get_whole_Corona_dataset():
    '''
    Instead of splitting the data into train and test data, it returns a whole
    preprocessed data.
    '''
    text = []
    Y = []
    i = 0
    nanTitle = 0
    nanText = 0
    cFile = coronafile
    breakI = splitSize 

    for index, d in cFile.iterrows():
        ftext = d['text']   # keep only the text and label
        ftitle = d['title']
        label = (d['label']).lower()
        
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

    # create an instance of a CountVectorizer, using 
    # (1) the standard 'english' stopword set from nltk, but lemmetized
    # (2) only keeping terms in the vocabulary that occur in at least 1% of documents
    # (3) allowing both unigrams and bigrams in the vocabulary (use "ngram_range=(1,2)" to do this)
    vectorizerText = CountVectorizer(stop_words = pf.getLemmatizedStopwords(), min_df=.01, ngram_range=(1,2), tokenizer= pf.LemmaTokenizer() )
    # create a sparse BOW array from 'text' using vectorizer  
    X = vectorizerText.fit_transform(text)
    
    #print('Vocabulary for text: ', vectorizerText.get_feature_names())

    return X, Y, vectorizerText


if __name__ == "__main__":
    X, Y = getCoronaText()
    getCoronaVocabulary(True)
    lt =  pf.LemmaTokenizer()
    print(lt(X[0]))

