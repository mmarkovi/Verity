# Verity
Classifying Misinformation in News project for CS 175 at University of California, Irvine, Winter 2021.

#### Project Title: 
Classifying Misinformation

#### Team: 
Misinformation-Verity

#### List of Team Members: 
Mia Markovic, Connor Couture, Justin Kang

### Project Summary
This project will use recurrent neural networks to predict whether the article is fake or real using general news and COVID-19 news datasets and also classify what broad category (like coronavirus, vaccines, etc.) that the misinformation falls into (a link to a factual source regarding this topic will be given). We plan to evaluate the models with cross-validation and also make a website where one can input a URL to a news article and the website will run it through our AI and both output whether it is fake or not and if it is misinformation it will be classified into one of our broad misinformation categories and give a link to a factual source about that topic. 

### Datasets
 - corona_fake : csv for our entire COVID-19 dataset, can be found at raw.githubusercontent.com/susanli2016/NLP-with-Python/master/data/corona_fake.csv
 - liar_test and liar_train: csv for testing and training data for the FakeNewsNet dataset.
 - fnn_test and fnn_train: csv for testing and training data for the FakeNewsNet dataset.
*Note: liar_test and fnn_test are too large to be uploaded. Files can be found at https://ieee-dataport.org/open-access/fnid-fake-news-inference-dataset#files*

### Preprocessing
**.py files**
 - preprocessingFunctions: various functions for preprocessing; Remove commas in between numbers in a string (replaceCommas), lemmatizer and tokenizer for CountVectorizer (LemmaTokenizer), and applying a CountVectorizer to a single string (getTermMatrixTestData).
 - covidPreprocess: provides functions for getting list of titles and text combined into one string and list of integer (0 or 1) labels (getCoronaText), or both these lists along with a CountVectorizer (getCoronaVocabulary, get_whole_Corona_dataset), with options for getting training or testing data for the COVID-19 dataset.
 - liarPreprocess: provides functions for getting list of titles and text combined into one string and list of integer (0 or 1) labels (getLiarText), or both these lists along with a CountVectorizer (getLiarVocabulary), with options for getting training or testing data for the LIAR dataset.
 - fnnPreprocess: provides functions for getting list of titles and text combined into one string and list of integer (0 or 1) labels (getFNNText), or both these lists along with a CountVectorizer (getFNNVocabulary), with options for getting training or testing data for the FakeNewsNet dataset.
 - combineMultipleDatasets: provides funtions for getting list of titles and text combined into one string and list of integer (0 or 1) labels (getAllText, getAllText2), or both these lists along with a CountVectorizer (getAllVocabulary), with options for getting training or testing data for the all three datasets combined.

*Note: all getVocabulary and getText functions were written with assistance from Assignment 1 for CS 175*

**.ipynb files**
 - DataVisualization: interactive python notebook which displays top 10 words in false/true articles and what percentage of the total words they are (for both fake/true only, or fake and true together), as well as the top 10 words that appear in true articles, but not false, and vice versa.

### Models
**.py files**
 - simpleModel: contains a simple feed forward neural network (SimpleNeuralNet) which can be trained and tested on either one of our datasets (trainAndTestSimpleModel).

*Note: SimpleNeuralNet was built with assistance from https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/01-basics/feedforward_neural_network/main.py#L37-L49 and trainAndTestSimpleModel was built with assistance from https://medium.com/analytics-vidhya/part-1-sentiment-analysis-in-pytorch-82b35edb40b8 and base code from assignment 2.*
 - saveModel: contains functions for saving our neural network models and corresponding vectorizers to a pickle file (saveModel, train_and_save_simple_model) and for loading out of a pickle file (loadModel), along with some test functions to make sure the model was still working after loading (testDataset, testText).

*Note: saving and loading files built with assistance from https://stackabuse.com/scikit-learn-save-and-restore-models/*
 - model_builder: repeated function for training of the model (train_simple_model_with_data), as well as a function for printing out accuracy, and other similar stats (train_and_save_vec_model).

**.ipynb files**
 - logisticClassifier:
 - CovidGeneralClassifier:
 - ModelAnalysis:
 - simpleModel:
 - simpleModelA:
 - simpleModelOneNeuronOutput:
 #### BERT
 **Folders**
  - corona: pickle files containing preprocessed, tokenized data from the ALBERTTokenizer for the COVID-19 dataset.
  - fnn: pickle files containing preprocessed, tokenized data from the ALBERTTokenizer for the FNN dataset.
 
 **.py files**
  - BERTPreprocess: contains a copied version of replaceCommas for ease of access, as well as functions for loading, tokenizing (tokenize_texts), and saving data to pickle files (load_fnn_data, load_corona_data, save_tokens_labels, load_tokens_labels).
  - BERTModel: contains a BERT and ALBERT model (BertBinaryClassifier, ALBERTModel) as well as two different options for fine tuning the ALBERT model, itself (load_and_process_data, fine_tune_albert).

*Note: BertBinaryClassifier, ALBERTModel, and load_and_process data were built using help from https://towardsdatascience.com/bert-to-the-rescue-17671379687f and https://github.com/LydiaXiaohongLi/Albert_Finetune_with_Pretrain_on_Custom_Corpus/blob/master/Albert_Finetune_with_Pretrain_on_Custom_Corpus_ToyModel.ipynb while fine_tune_albert was built with assistance from https://medium.com/@aniruddha.choudhury94/part-2-bert-fine-tuning-tutorial-with-pytorch-for-text-classification-on-the-corpus-of-linguistic-18057ce330e1*
  - logisticNN: contains a simple neural network (LogisticBinaryClassifier) for use in testing the preprocessed BERT inputs saved from BERTPreprocess, as well as functions to train (train_model) and save the finished model (save_log_model).
  - validation: contains functions for training (validate_LogisticNN) and testing the accuracy of the model from logisticNN using testing data (get_model_accuracy)
 
 
### Validation
**.ipynb files**
 - validation: file for testing accuracy of simple neural network (train_simple_model_with_data) and with various functions for printing out accuracy and other stats (get_model_accuracy, chart_epoch_diff, chart_num_layer_diff).


### Web-app
 - 
 
