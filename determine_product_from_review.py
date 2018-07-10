#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 15:53:28 2018

@author: ambikaajaggi
"""

# =============================================================================
# This code has been written using:
# https://towardsdatascience.com/multi-class-text-classification-with-scikit-
# learn-12f1e60e0a9f
# =============================================================================

import os
import pandas as pd

cwd = os.getcwd()  # Gets the current working directory (cwd)
files = os.listdir(cwd)  # Gets all the files in that directory

df=pd.read_csv('GrammarandProductReviews.csv')
col = ['name', 'reviews.text']
df = df[col]

n = len(df['name'])

# create the product ID as a unique integer
df['product_id'] = df['name'].factorize()[0]
product_id_df = (df[['name', 'product_id']].drop_duplicates()
                                   .sort_values('product_id'))
product_to_id = dict(product_id_df.values)
id_to_product = dict(product_id_df[['product_id', 'name']].values)

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=2, norm='l2', 
                        encoding='latin-1', ngram_range=(1, 2), 
                        stop_words='english')
features = tfidf.fit_transform((df['reviews.text']).values.astype('U'))
labels = df.product_id

from sklearn.feature_selection import chi2
import numpy as np
N = 2
for Product, product_id in sorted(product_to_id.items()):
  features_chi2 = chi2(features, labels == product_id)
  indices = np.argsort(features_chi2[0])
  feature_names = np.array(tfidf.get_feature_names())[indices]
  unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
  bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
  #print("# '{}':".format(Product))
  #print(" . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-N:])))
  
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import pickle

X_train, X_test, y_train, y_test = train_test_split(df['reviews.text'],
                                                    df['name'], 
                                                    random_state = 0)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train.values.astype('U'))

save_vectoriser = open("CountVectorizer.pickle","wb")
pickle.dump(count_vect, save_vectoriser)
save_vectoriser.close()

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
clf = MultinomialNB().fit(X_train_tfidf, y_train)

save_classifier1 = open("MultinomialNB.pickle","wb")
pickle.dump(clf, save_classifier1)
save_classifier1.close()

#example of how to use:
#print(clf.predict(count_vect.transform(
#["This is the best cream I have ever used!!!"])))