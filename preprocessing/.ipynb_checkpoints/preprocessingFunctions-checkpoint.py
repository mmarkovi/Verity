#Written by Mia Markovic (mmarkovi, 39425669) 
 
# ---------------------------------------------------------------------------------------
# CS 175, WINTER 2021: DATA PREPROCESSING SHARED FUNCTIONS
#
# ---------------------------------------------------------------------------------------
import re
from nltk import word_tokenize
import string
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords
#nltk.download('wordnet') #had to import this in order for the above to work

lemmatizer = WordNetLemmatizer()
lemmStop = [lemmatizer.lemmatize(t) for t in stopwords.words('english')]
lemmStop += ['could', 'might', 'must', 'need', 'sha', 'wo', 'would']

def getLemmatizedStopwords():
    return lemmStop

def replaceCommas(strToRepl):
    '''function to search for numbers like 100,000 and replace them with 100000
        so that this number will stay combined when we perform tokenization
    '''
    #searches to see if there is [digit,digit] in the text
    a = re.search(r'[0-9],[0-9]', strToRepl) 
    while (a != None): #if there is no more matches, then a will be None
        b = a.span()[0] + 1 #second character will be a comma (according to how we searched for it)
        strToRepl = strToRepl[:b] +  strToRepl[b+1:] #take everything but the comma
        a = re.search(r'[0-9],[0-9]', strToRepl)
    return strToRepl

#written based off of code found in 
# https://scikit-learn.org/stable/modules/feature_extraction.html
# under section 6.2.3.10
class LemmaTokenizer:
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        
    def __call__(self, text):
        return [self.lemmatizer.lemmatize(t) for t in word_tokenize(text) if t.isalnum()]

