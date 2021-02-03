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


### Technical Approach
For classifying whether a document is misinformation, we will start by combining the coronavirus misinformation dataset and the general fake news misinformation dataset into a larger dataset that will be then processed. For the base model, we will parse the texts into bags of words for training the model. Then, we will make a simple neural network with a sigmoid activation function and cross-entropy loss function to build a model for our classifier. The number of hidden layers will be determined as we build models with different numbers of them and check for overfitting and underfitting with cross-validation. For the advanced model, we will use BERT (embedding tool) to process the text along with other metadata such as dates and author to train our model. The model will use the reinforced neural network with a ReLU activation function and a cross-entropy loss function. Then, same as what we will do on the base model, we will try to fit the data properly by changing the number of hidden layers with the help of cross-validation and visualization of the accuracy.

Classifying text classified as misinformation into broader categories is more of a stretch goal. The plan would be to use supervised topic classification that would involve making a topic classification model that is trained on misinformation categorized into several large broad categories of our choosing that has factual information easily accessible (such as coronavirus, vaccination, etc.). For this supervised learning task, we plan on starting with simpler methods first, such as Naïve Bayes and Logistic Regression, and then if they are not sufficiently good at doing the classification task well, we would try using neural networks for this task, however, we hope to avoid the use of neural networks so we don’t need as much data for good model performance. We intend to use a hand-labeled dataset of misinformation categorized into categories of our choosing. The plan is to take datasets that already are for misinformation classification, remove the entries from the datasets that are labeled to be true information (so we can ignore the true/fake label), and then add a column to the dataset with the topic of the entry. Like the model for classifying whether the given text is misinformation or not, this model we will initially train through having the text being parsed into bag of words, and then if the performance using bag of words is not satisfactory, or we have extra time, we will recreate the model using embeddings. Like the misinformation classification model, we plan on testing and evaluating this model with cross-validation.

While we are building and evaluating the model, we will also make a web application. We will use the web application to get user input and predict whether it is fake or not, and if it’s fake, the topic it pertains to will be listed and a link for factual information that relates to the topic will be given. As it is not a complicated process to get text, it will only contain a plain text form and submit button. If the time permits, we will also allow users to give the web url of an article to use as input. For this frontend, we plan on using Vue.js framework along with Sass styling tool to make an interactive and clean website. For server side application, we will use Flask Restful API to send the user input to the server and process the input with our Python model.


### Data Sets
We are planning on using at least two misinformation data sets to check for fake news. The first being a generalized fake news article dataset, and the second being a COVID-19 specific misinformation dataset. We will use data from https://towardsdatascience.com/explore-covid-19-infodemic-2d1ceaae2306 for our COVID-19 data, and https://ieee-dataport.org/open-access/fnid-fake-news-inference-dataset#files for our generalized news data. The COVID-19 data set contains 586 true articles and 578 fake ones with the website’s text, which domain it originated from, and whether the data is true or fake. The regular news articles’ dataset contains the article’s data, speaker, which websites the data was found, the text, and whether the article is true or fake. This dataset contains over 16,000 articles of varying categories. The data we are using is not already tokenized, as all the data we have found so far has not been preprocessed yet. However, while we will start with just the raw document data to start out with, we hope to expand to other data too, like domain, author, or date to try and classify our data.


### Experiments and Evaluation
We will use cross-validation to evaluate our model. Simple train/test split with AUC scores can give reasonably good results, but there is a good chance of forming a bias because the test dataset can have more fake information than train dataset and vice-versa. So, we will use stratified k-fold cross-validation. It is a simple data splitting method. We will first shuffle the dataset and split them into k groups. Each fold (group) will have the same (or similar) false-to-true ratio so that we can have an unbiased dataset in each iteration. Then, we will calculate AUC as our evaluation metric on each fold and take an average of them to assess how overfitting our model is. This method will be used over multiple models to compare different parameters and techniques. We will tune some parameters based on this evaluation metric and build a good fit model for a final version.
To assess the quality of our model, we will also gather false positive results. Since the classifiers can give a conditional probability of labeling an article as fake given that it is real, we can use it to rank the articles, find what kind of errors there are, and get an idea of what heuristics we can add to reduce this probability. This will also be useful for finding what kind of words make the classifier think the article is fake for our final report.


### Software 
We plan on mostly writing our own code for this project. We will create a GitHub repository to coordinate our code development. We will write the majority of the logic on our own. 

In terms of using outside resources, we plan on using code we have found in the GitHub projects listed on piazza, the code used in the datasets above, or other software we find online. The public code/software/libraries we use will likely be for preprocessing, model building, and web development. We might also follow techniques to try and analyze our data, like those described in one of these articles: https://towardsdatascience.com/identifying-fake-news-the-liar-dataset-713eca8af6ac, https://towardsdatascience.com/fake-news-classification-with-recurrent-convolutional-neural-networks-4a081ff69f1a. We will probably use these articles as inspiration for code we may write.
#### Public Code
##### Coding Languages:
JavaScript - for website, Python 3.8 - for everything else in project

##### Libraries and Frameworks:
NumPy, SciPy, NLTK, Matplotlib, PyTorch, Scikit-learn, Vue.js, Flask, Sass

#### Code We Will Make
##### Top Words for Classification:
Calculates and prints the top weighted words associated with classifying a document as misinformation or as being true. Calculates and prints the top weighted words associated with categorizing misinformation into each topic category.

##### Website
Styles the website with HTML/JavaScript/Sass(CSS) codes. Handles HTTP Request with Flask Restful API


### Milestones
Weeks 4-5: Gathering datasets, starting to work on tokenizing functions and other preprocessing steps for the files. Start to analyze data and see possible trends we could use to train our models on. Mock up the design of our website/UI and what features we want the user to be able to interact with.

Weeks 6-8: Start training models on data using the initial trends discovered. See initial model results on training and test data and work to combat overfitting and increase accuracy for test data. Start working on website/UI in order for inputting new data and start implementing some advanced methods, if time permits.

Weeks 9-10: Work on implementing more advanced methods and compare performance to the basic methods. Check to see if we should switch to using all advanced methods or just taking certain parts that increase the precision. Finish working on user interface and implement classification methods, if time permits.

### Individual Student Responsibilities
Mia Markovic: will focus primarily on the preprocessing steps. Will initially work with Connor to gather the datasets. Then, will use various functions, like word_tokenize, pos_tag, and FreqDist from NLTK in order to separate the data. Will also do some sort of tokenizing/organizing for other features, like website domain or authors for extra parameters to use for hyperparameter tuning. After finishing with the preprocessing steps, will work with Connor on building a model for classifying the data.

Connor Couture: will focus primarily on the model building steps for classifying the data. Initially, will work with Mia on gathering data and then work on building and tuning the models. Will work on visualizing the data for the project reports and for determining which parameters and models to use. 

Justin Kang: will mainly work on validation testing and creating the web application. Will use Vue.js and Flask Restful API to work on building a website for the project interface and connecting our prediction model to the website. Once the model is decently developed, will work with Connor to tune the model with cross validation testing. 
