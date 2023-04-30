## Project Overview

This project contains two versions of the Twitter Sentiment Analysis:

Version 1: A Logistic Regression model was trained using an average CountVectorizer, achieving an accuracy of 90%
Version 2: A deep learning model was trained using a Long Short-Term Memory (LSTM) network with a Sequential model in TensorFlow and Keras, achieving an accuracy of 93%.

# Twitter Sentiment Analysis

This project aims to analyze the sentiment of tweets from Twitter using machine learning and deep learning techniques. The dataset used for training and testing the models is a collection of 1.6 million tweets, containing equal numbers of positive and negative tweets.

## Usage

To use this project, you will need to have Python 3 installed, along with the following libraries: NumPy, Pandas, NLTK, Scikit-learn, TensorFlow, and Keras. Clone the repository and run the Jupyter Notebook files to train and test the models.

## Dataset

The dataset used in this project is available in kaggle, which contains thousands of tweets. Each tweet is labeled as either positive or negative, and the dataset has been preprocessed and cleaned to remove any irrelevant or sensitive information.

## Models

Two models were trained and tested in this project. In the first version, a Logistic Regression model was trained using an average CountVectorizer, achieving an accuracy of 90%. In the second version, a deep learning model was trained using a Long Short-Term Memory (LSTM) network with a Sequential model in TensorFlow and Keras, achieving an accuracy of 93%.

## Future Work

Future work on this project could involve exploring other deep learning architectures for sentiment analysis, such as Convolutional Neural Networks (CNNs) or Attention Mechanisms. Additionally, more data could be collected and annotated to improve the accuracy of the models. Finally, the models could be deployed as a web service to provide real-time sentiment analysis on live Twitter data.
