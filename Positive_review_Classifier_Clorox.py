#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 10:13:18 2018

@author: ambikaajaggi
"""

#Negative Review Classifier

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer as sid
import os
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from gensim.models.word2vec import Word2Vec
import string

cwd = os.getcwd()  # Gets the current working directory (cwd)
files = os.listdir(cwd)  # Gets all the files in that directory

df=pd.read_csv('GrammarandProductReviews.csv')

n = len(df['reviews.rating'])

# Create the dataframe which only contains real customers for Clorox wipes
positive_reviews = {}
pos = 0

negative_reviews = {}
neg = 0

for i in range(n):
    if (df['reviews.didPurchase'][i]!=False) and (df['name'][i]==
       'Clorox Disinfecting Wipes Value Pack Scented 150 Ct Total'):
            if (df['reviews.rating'][i] > 3):
                positive_reviews[pos] = str(df['reviews.text'][i])
                pos += 1
            else:
                negative_reviews[neg] = str(df['reviews.text'][i])
                neg += 1

# removing stop words and punctuation
# credit to: 
# https://dzone.com/articles/opinion-mining-python-implementation  
def preProcessing(input_data):
    cachedStopWords = nltk.corpus.stopwords.words("english")
    for i in range(len(input_data)):
        sentence = []
        sent = nltk.word_tokenize(input_data[i])
        for w in sent:
            if w not in cachedStopWords and w not in string.punctuation:
                sentence.append(w)
        input_data[i] = sentence
    return input_data

preProcessing(negative_reviews)
preProcessing(positive_reviews)

# Create a POS tagger and aspect extractor that will be used for aspect 
# extraction, credit to: 
# https://dzone.com/articles/opinion-mining-python-implementation         
def posTagging(input_data):
        output_data = {}
        for i in range(len(input_data)):
            output_data[i]=nltk.pos_tag((input_data[i]))     
        return output_data

def aspectExtraction(input_data):
    prevWord=''
    prevTag=''
    currWord=''
    aspectList=[]
    outputDict={}
    input_data = posTagging(input_data)
    #Extracting Aspects
    for key,value in input_data.items():
        for word,tag in value:
            if(tag=='NN' or tag=='NNP'):
                if(prevTag=='NN' or prevTag=='NNP'):
                    currWord= prevWord + ' ' + word
                else:
                    aspectList.append(prevWord.upper())
                    currWord= word
            prevWord=currWord
            prevTag=tag
    #Eliminating aspect which has 1 or less count
    for aspect in aspectList:
            if(aspectList.count(aspect)>1):
                    if(outputDict.keys()!=aspect):
                            outputDict[aspect]=aspectList.count(aspect)
    outputAspect=sorted(outputDict.items(), key=lambda x: x[1],reverse = True)
    
    
    return outputAspect

negative_reviews_aspects = aspectExtraction(negative_reviews)
positive_reviews_aspects = aspectExtraction(positive_reviews)    

# unique features to the positive list
positive_reviews_aspects_new = list(set(positive_reviews_aspects
                                        ).difference(negative_reviews_aspects))

positive_reviews_aspects_indices = []
senti_analyzer = sid()

# filter out any features which are negative by storing all negative 
# review indices
for i in range(len(positive_reviews_aspects_new)): 
    if senti_analyzer.polarity_scores(
            positive_reviews_aspects_new[i][0])['compound']<=0:
        positive_reviews_aspects_indices.append(i)

positive_reviews_unique_features = [v for i, v in enumerate(
                                    positive_reviews_aspects) if i not in 
                                    positive_reviews_aspects_indices]

positive_reviews_features_list = []
for i in range(len(positive_reviews_unique_features)):
    positive_reviews_features_list.append(
            positive_reviews_unique_features[i][0])

useful_feedback = []

# get all reviews which contain important features
for i in range(len(positive_reviews)):
    index = 0
    for feature in positive_reviews_features_list:
        if feature in positive_reviews[i]:
            useful_feedback.append(positive_reviews[i])
        index += 1
           
# create the useful feedback intoa form that can be used for the Word2Vec model
w2v_sentences = []
       
for i in range(len(useful_feedback)):
    if useful_feedback[i] != []:
        w2v_sentences.append(useful_feedback[i])

for i in range(len(w2v_sentences)):
    w2v_sentences[i] = ' '.join(map(str, w2v_sentences[i]))

w2v_formatted_sentences = []
for i in range(len(w2v_sentences)):
    w2v_formatted_sentences.append(w2v_sentences[i].split())
    
positive_reviews_features_formatted_sentences = []
    
for i in range(len(positive_reviews_features_list)):
    positive_reviews_features_formatted_sentences.append(
            positive_reviews_features_list[i].split())

model_words = Word2Vec(sentences=positive_reviews_features_formatted_sentences, 
                       size=100, window=5, min_count=1, workers=4, sg=0)

vocab = list(model_words.wv.vocab)
X = model_words[vocab]

pca = PCA(n_components=2)
result = pca.fit_transform(X)

plt.scatter(result[:, 0], result[:, 1])
words = list(model_words.wv.vocab)
for i, word in enumerate(words):
	plt.annotate(word, xy=(result[i, 0], result[i, 1]))
plt.xlim([-0.012, 0.015])
plt.ylim([-0.015, 0.02])
plt.figure(figsize=(200000,100000))
plt.show()