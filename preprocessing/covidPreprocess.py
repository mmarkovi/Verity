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

coronafile =  pd.read_csv("../datasets/corona_fake.csv")

#cleaning up broken data according to labels added by
# https://towardsdatascience.com/explore-covid-19-infodemic-2d1ceaae2306
coronafile.loc[5]['label'] = 'fake'
coronafile.loc[15]['label'] = 'true'
coronafile.loc[43]['label'] = 'fake'
coronafile.loc[131]['label'] = 'true'
coronafile.loc[242]['label'] = 'fake'


def getCoronaVocabulary():
    text = []
    #titles = []
    Y = []
    i = 0
    nanTitle = 0
    nanText = 0
    print('\nExtracting tokens from each review.....(can be slow for a large number of reviews)......')   
    for d in coronafile.loc:
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
        #titles.append(ftitle)
        Y.append(score)
        i += 1
        if (i == 1164):
            #for some reason the for loop doesnt know when to stop so put in a manual break
            break
    print("there are", nanTitle, "nan titles")
    print("there are", nanText, "nan text")
        
    # create an instance of a CountVectorizer, using 
    # (1) the standard 'english' stopword set 
    # (2) only keeping terms in the vocabulary that occur in at least 1% of documents
    # (3) allowing both unigrams and bigrams in the vocabulary (use "ngram_range=(1,2)" to do this)
    vectorizerText = CountVectorizer(stop_words = 'english', min_df=.01, ngram_range=(1,2), tokenizer= pf.LemmaTokenizer() )
    #vectorizerNoLem = CountVectorizer(stop_words = 'english', min_df=.01, ngram_range=(1,2)) #no lemmatization
    
    # create a sparse BOW array from 'text' using vectorizer  
    X = vectorizerText.fit_transform(text)
    #X2 = vectorizerNoLem.fit_transform(text)
    
    print('Data shape for text: ', X.shape)
    p#rint('Data shape for text: ', X2.shape)
    
    #can comment out to not see the vocabularies
    print('Vocabulary for text: ', vectorizerText.get_feature_names())

    return X, Y, vectorizerText

if __name__ == "__main__":
    getCoronaVocabulary()

# preprocessing

# second dataset: convert the truthfulness into binary variable -- DONE
# remove the data entry with NaN text and NaN title  -- DONE
# combine the text and title as a single feature  -- DONE
# lemmatization --DONE
# try to get the whole number (e.g. 100,000 is counted as "100,000" instead of "100","000")  -- DONE
# trigram (for future implementation) -easy to add later
# separate the sources into names type and url types (for future implementation) -need to check in 
