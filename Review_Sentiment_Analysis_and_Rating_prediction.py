#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 10:17:24 2018

@author: ambikaajaggi
"""
from nltk.tokenize import word_tokenize
from nltk.corpus import words
import string
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer as sid
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

cwd = os.getcwd()  # Gets the current working directory (cwd)
files = os.listdir(cwd)  # Gets all the files in that directory

df=pd.read_csv('GrammarandProductReviews.csv')

n = len(df['reviews.rating'])

# =============================================================================
# Create dictionaries of the reviews and ratings, save them as pickle files
# =============================================================================

d_review = {}
d_rating = {}
sentiment = {}
negative_reviews = {}
positive_reviews = {}
neg = 0
pos = 0

k = 0

for i in range(n):
    if df['reviews.didPurchase'][i]!=False:
        d_review[k] = df['reviews.text'][i]
        d_review[k] = str(d_review[k]).lower()
        d_rating[k] = df['reviews.rating'][i]
        if df['reviews.rating'][i] < 4:
            negative_reviews[neg] = str(df['reviews.text'][i])
            neg += 1
        else:
            positive_reviews[pos] = str(df['reviews.text'][i])
            pos += 1
        k += 1
        
save_review_data = open("Review.pickle","wb")
pickle.dump(d_review, save_review_data)
save_review_data.close()

save_Nreview_data = open("NegReview.pickle","wb")
pickle.dump(negative_reviews, save_Nreview_data)
save_Nreview_data.close()

save_Preview_data = open("PosReview.pickle","wb")
pickle.dump(positive_reviews, save_Preview_data)
save_Preview_data.close()

save_rating_data = open("Rating.pickle","wb")
pickle.dump(d_rating, save_rating_data)
save_rating_data.close()

# =============================================================================
# Filter out stop-words, punctuation and misspelled words and perform sentiment 
# analysis to find the compound sentiment of the review. Save the sentiments as 
# a pickle file
# =============================================================================

stop_words = set(stopwords.words("english"))
word_list = words.words()
senti_analyzer = sid()

m = len(d_review)

for j in range(m):
    sent = []
    word = word_tokenize(d_review[j])
    for w in word:
        if w not in stop_words and w not in string.punctuation \
        and w in word_list:
            sent.append(w)
    sent = ' '.join(sent)
    ss = senti_analyzer.polarity_scores(sent)
    sentiment[j] = ss['compound'] # compound of neg, pos and neutral sentiment
    print("Sentiment score for review number {0}: {1} is {2}".format((j+1), 
          d_review[j], sentiment[j]))

save_sentiment_data = open("Sentiment.pickle","wb")
pickle.dump(sentiment, save_sentiment_data)
save_sentiment_data.close()

# =============================================================================
# Open the pickle files for sentiment and rating, using NumPy create the X and
# Y variables into a matrix format that can be split into training and testing 
# sets and inputted into the classifiers
# =============================================================================

sentiment_f = open("Sentiment.pickle","rb")
sentiment = pickle.load(sentiment_f)
sentiment_f.close()

rating_f = open("Rating.pickle","rb")
rating = pickle.load(rating_f)
rating_f.close()

X = (np.fromiter(sentiment.values(), dtype=float)).reshape(-1,1)
Y = (np.fromiter(rating.values(), dtype=int))
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.10, 
                                                    random_state=46)

# =============================================================================
# Train and test classifiers, pickling any for future use
# =============================================================================

classifier1_f = open("LinearSVC.pickle","rb")
classifier1 = pickle.load(classifier1_f)
classifier1_f.close()

preds = classifier1.predict(X_test)
print(list(preds[:10]))
print(list(y_test[:10]))

from sklearn.svm import LinearSVC

classifier1 = LinearSVC()

classifier1.fit(X_train, y_train)

preds = classifier1.predict(X_test)

print(list(preds[:10]))
print(list(y_test[:10]))

print('LinearSVC:', accuracy_score(y_test, preds))

save_classifier1 = open("LinearSVC.pickle","wb")
pickle.dump(classifier1, save_classifier1)
save_classifier1.close()

from sklearn.linear_model import SGDClassifier

classifier2 = SGDClassifier(max_iter=100000)
classifier2.fit(X_train, y_train)
preds = classifier2.predict(X_test)

print(list(preds))
print(list(y_test))

print('SGDClassifier:', accuracy_score(y_test, preds))

save_classifier2 = open("SGD.pickle","wb")
pickle.dump(classifier2, save_classifier2)
save_classifier2.close()

#classifier2_f = open("SGD.pickle","rb")
#classifier2 = pickle.load(classifier2_f)
#classifier2_f.close()
#preds = classifier2.predict(X_test)
#
#print(list(preds[:20]))
#print(list(y_test[:20]))
#
#print('SGDClassifier:', accuracy_score(y_test, preds))
#
#x = np.reshape(y_test-preds, (-1, 2))

from sklearn.naive_bayes import GaussianNB

classifier3 = GaussianNB()
classifier3.fit(X_train, y_train)
preds = classifier3.predict(X_test)

print(list(preds[:20]))
print(list(y_test[:20]))

print('GaussianNB:', accuracy_score(y_test, preds))

save_classifier3 = open("GaussianNB.pickle","wb")
pickle.dump(classifier3, save_classifier3)
save_classifier3.close()

from sklearn.linear_model import LogisticRegression

# fit a logistic regression model to the data
model = LogisticRegression(tol = 0.00001, max_iter = 1000)
model.fit(X_train, y_train)
# make predictions
preds = model.predict(X_test)
# summarize the fit of the model
print('LogisticRegression:', accuracy_score(y_test, preds))

from sklearn.svm import SVC
classifier4 = SVC()
classifier4.fit(X_train, y_train)
preds = classifier4.predict(X_test)

print('SVC:', accuracy_score(y_test, preds))

save_classifier4 = open("SVC.pickle","wb")
pickle.dump(classifier4, save_classifier4)
save_classifier4.close()