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
    titles = []
    Y = []
    lengths = []
    i = 0
    nanTitle = 0
    nanText = 0
    print('\nExtracting tokens from each review.....(can be slow for a large number of reviews)......')   
    for d in coronafile.loc:
        ftext = d['text']     # keep only the text and label
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
            
        text.append(ftext)   
        titles.append(ftitle)
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
    vectorizerText = CountVectorizer(stop_words = 'english', min_df=.01, ngram_range=(1,2))
    vectorizerTitle = CountVectorizer(stop_words = 'english', min_df=.01, ngram_range=(1,2))
    
    # create a sparse BOW array from 'text' using vectorizer  
    X = vectorizerText.fit_transform(text)
    X2 = vectorizerTitle.fit_transform(titles)
    
    print('Data shape for text: ', X.shape)
    print('Data shape for titles: ', X.shape)
    
    #can comment out to not see the vocabularies
    print('Vocabulary for text: ', vectorizerText.get_feature_names())
    print()
    print("----------------------------------------------------------")
    print()
    print('Vocabulary for titles: ', vectorizerTitle.get_feature_names())

    return X, Y, vectorizerText, X2, vectorizerTitle

    # data visualization
    
    # top 10 words appeared in False articles (display a percentage of the word bank)
    # top 10 words appeared in True articles (display a percentage of the word bank)
    # top 10 words that appear in the True article and not appear in False article
    # top 10 words that appear in the False article and not appear in True article
    # Find the word with the most and least frequency
    # Find the True-to-False ratio on Nan title and NaN text (for finding whether removing them can cause a significant bias)

    # preprocessing

    # second dataset: convert the truthfulness into binary variable
    # remove the data entry with NaN text and NaN title
    # combine the text and title as a single feature
    # lamentization
    # try to get the whole number (e.g. 100,000 is counted as "100,000" instead of "100","000")
    # trigram (for future implementation)
    # separate the sources into names type and url types (for future implementation)

    # UI/UX
    # Figma for prototype (home page, about page)
    # Vue init file structure
    # user study with people you know (for future implementation)
    # user study to find how better humans can classify the false information than the model

getCoronaVocabulary()