# Verity
Classifying Misinformation in News project for CS 175 at University of California, Irvine, Winter 2021.

#### Project Title: 
Classifying Misinformation

#### Team: 
Misinformation-Verity

#### List of Team Members: 
Mia Markovic, Connor Couture, Justin Kang

### Project Summary
In this project, we are attempting to investigate the issue of misinformation and fake news in our media, and classify articles as true or fake using natural language processing. The project uses a combination of simple and deep feed-forward neural networks to classify articles as being general news or COVID-19 related, and then feeds this information into the corresponding classifier. This model takes in the preprocessed text, done with a bag of words, and then outputs a label. The models have been validated using cross-validation as well as a user study for analysis. The model will be fully deployed on our website, where one can input an article title and plaintext to find out whether it is true or fake. 


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
 - logisticClassifier: contains our first model, a simple logistic classifier, which we trained (logisticClassifyData) on various max_iter values to see if we could reach convergence, based on which bag of words model was passed in (getVocabulary).
 - simpleModel: contains updated code from simpleModel.py for the neural network (SimpleNeuralNet) and the training step (trainSimpleModel, train_simple_model_with_data) and testing step (testModel), along with tests for various epochs.
 - simpleModelA: contains updated code from simpleModel.py for the neural network (SimpleNeuralNet) and the training and testing steps (trainAndTestSimpleModel), along with tests for various epochs. Also contains the base model (TwoHiddenLayerNeuralNet) and training and testing steps (trainAndTestTwoHiddenLayerModel) for a two layer feed forward neural network.
 - simpleModelOneNeuronOutput: updated neural networks from simpleModelA (SimpleNeuralNet, TwoHiddenLayerNeuralNet) to have a single neuron output.
 - CovidGeneralClassifier: contains both a logistic classifier (logistic_classification) and a simple neural net (SimpleGeneralNeuralNet, trainSimpleModel) with various tests to determine which model whad higher testing accuracy.
 - ModelAnalysis: contains updated code from simpleModel.ipynb for the neural network (SimpleNeuralNet) and training the data (trainAndTestSimpleModelAndGetProbs), allowing for easy visualization of different documents based on output probabilities. 
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
 **Folders**
 - model: contains pickles of saved neural networks for each model and corresponding vectorizers
 - static:
 - templates:

**.py files**
 - preprocessingFunctions: copy of same file from the Preprocessing folder for use for the website.
 - model_lib: copy of the classes for the saved neural networks (SimpleNeuralNet, TwoHiddenLayerNeuralNet, SimpleGeneralNeuralNet), along with loading functions from the pickle (load_model, load_covid_model, load_fnn_model, load_general_model) and testing functions (covid_general_predict_model, predict_model) for the website.
 - app: contains the main functions used in the website for displaying and obtaining a prediction.

**other files**
 - yarn.lock:
 - package.json:
