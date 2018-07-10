#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 10:13:18 2018

@author: ambikaajaggi
"""

#Negative Review Classifier

import nltk
import os
import pandas as pd

cwd = os.getcwd()  # Gets the current working directory (cwd)
files = os.listdir(cwd)  # Gets all the files in that directory

df=pd.read_csv('GrammarandProductReviews.csv')

n = len(df['reviews.rating'])

# Create the dataframe which only contains real customers for Nexxus gel
positive_reviews = {}
pos = 0

negative_reviews = {}
neg = 0

for i in range(n):
    if (df['reviews.didPurchase'][i]!=False) and (df['name'][i]==
       'Nexxus Exxtra Gel Style Creation Sculptor'):
            if (df['reviews.rating'][i] > 3):
                positive_reviews[pos] = str(df['reviews.text'][i])
                pos += 1
            else:
                negative_reviews[neg] = str(df['reviews.text'][i])
                neg += 1

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

# unique features to the negative list
negative_reviews_unique_features = (list(set(negative_reviews_aspects)
                                    .difference(positive_reviews_aspects)))

negative_reviews_features_list = []
for i in range(len(negative_reviews_unique_features)):
    negative_reviews_features_list.append(
            negative_reviews_unique_features[i][0])

useful_feedback = []

# get all reviews which contain important features
for i in range(len(negative_reviews)):
    index = 0
    for feature in negative_reviews_features_list:
        if feature in negative_reviews[i]:
            useful_feedback.append(negative_reviews[i])
        index += 1

useful_feedback = list(set(useful_feedback))

import difflib

# only keep unique revews to minimize reading the same complain multiple times
unique_reviews = []
unique_reviews.append(useful_feedback[0].lower())
for i in range(len(useful_feedback)-1):
    for j in range(i, len(useful_feedback), 1):
        test1 = useful_feedback[i].lower()
        test2 = useful_feedback[j].lower()
        if ((difflib.SequenceMatcher(None, test1, test2)).ratio()<0.0005 and
        useful_feedback[j].lower() not in unique_reviews):
           unique_reviews.append(useful_feedback[j].lower())